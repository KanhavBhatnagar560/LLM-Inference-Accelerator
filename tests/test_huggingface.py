import unittest
from types import SimpleNamespace
from unittest import mock

from specdecode.huggingface import HuggingFaceCausalLM, HuggingFaceModelPair
from specdecode.tokenizers import TokenizerCompatibilityError


class FakeInput:
    def __init__(self, values):
        self.values = values
        self.shape = (len(values), len(values[0]))
        self.dtype = "long"


class FakeSelected:
    def __init__(self, rows):
        self.rows = rows

    def float(self):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.rows


class FakeLogits:
    def __init__(self, rows):
        self.rows = rows
        self.ndim = 3
        self.shape = (1, len(rows), len(rows[0]))

    def __getitem__(self, key):
        batch, positions, vocabulary = key
        assert batch == 0
        assert vocabulary == slice(None)
        return FakeSelected([self.rows[position] for position in positions])


class FakeInferenceMode:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class FakeTorch:
    long = "long"

    def tensor(self, values, **kwargs):
        return FakeInput(values)

    def ones_like(self, value):
        return FakeInput([[1 for _ in value.values[0]]])

    def ones(self, shape, **kwargs):
        return FakeInput([[1 for _ in range(shape[1])] for _ in range(shape[0])])

    def inference_mode(self):
        return FakeInferenceMode()

    def softmax(self, selected, dim):
        return selected


class FakeCudaTask:
    def __init__(self, result):
        self.result = result

    def wait(self):
        return self.result


class FakeCudaRuntime:
    def __init__(self):
        self.transfers = []
        self.submissions = []

    def copy_to_device(self, name, source):
        task = FakeCudaTask(source)
        self.transfers.append((name, source, task))
        return task

    def submit_draft(self, operation, *, wait_for, label):
        self.submissions.append(("draft", wait_for, label))
        return FakeCudaTask(operation())

    def submit_target(self, operation, *, wait_for, label):
        self.submissions.append(("target", wait_for, label))
        return FakeCudaTask(operation())


class FakePagedCacheMirror:
    def __init__(self):
        self.synchronized = []
        self.truncated = []
        self.resets = 0

    def synchronize(self, cache):
        self.synchronized.append(cache)

    def truncate(self, length):
        self.truncated.append(length)

    def reset(self):
        self.resets += 1


class FakeTokenizer:
    bos_token_id = None
    eos_token_id = None
    pad_token_id = None
    unk_token_id = None
    additional_special_tokens_ids = []

    def __init__(self, vocabulary=None):
        self.vocabulary = vocabulary or {"a": 0, "b": 1}

    def __len__(self):
        return len(self.vocabulary)

    def get_vocab(self):
        return dict(self.vocabulary)

    def get_added_vocab(self):
        return {}


class FakeAutoTokenizer:
    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        if model_id == "draft":
            return FakeTokenizer({"a": 0, "b": 1})
        return FakeTokenizer({"a": 1, "b": 0})


class FakeModel:
    config = SimpleNamespace(vocab_size=2)

    def __init__(self):
        self.eval_called = False
        self.calls = []
        weight = SimpleNamespace(device="cpu")
        self.embeddings = SimpleNamespace(num_embeddings=2, weight=weight)

    def eval(self):
        self.eval_called = True

    def get_input_embeddings(self):
        return self.embeddings

    def __call__(self, *, input_ids, attention_mask, use_cache):
        tokens = list(input_ids.values[0])
        self.calls.append((tokens, use_cache))
        rows = [[0.75, 0.25] if token == 0 else [0.25, 0.75] for token in tokens]
        return SimpleNamespace(logits=FakeLogits(rows))


class FakeCache:
    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.crop_calls = []

    def crop(self, length):
        self.crop_calls.append(length)
        self.tokens = self.tokens[:length]


class FakeWindowCache(FakeCache):
    def get_seq_length(self):
        return len(self.tokens)


class FakeLegacyKV:
    ndim = 4

    def __init__(self, sequence_length):
        self.sequence_length = sequence_length

    def __getitem__(self, key):
        ellipsis, sequence, final = key
        assert ellipsis is Ellipsis
        assert final == slice(None)
        return FakeLegacyKV(sequence.stop)


class FakeCachedModel(FakeModel):
    def __call__(
        self,
        *,
        input_ids,
        attention_mask,
        use_cache,
        past_key_values=None,
    ):
        tokens = list(input_ids.values[0])
        previous = [] if past_key_values is None else list(past_key_values.tokens)
        self.calls.append(
            {
                "input_tokens": tokens,
                "attention_length": len(attention_mask.values[0]),
                "past_length": len(previous),
                "use_cache": use_cache,
            }
        )
        rows = []
        for index in range(len(tokens)):
            prefix_length = len(previous) + index + 1
            probability = prefix_length / 10.0
            rows.append([probability, 1.0 - probability])
        return SimpleNamespace(
            logits=FakeLogits(rows),
            past_key_values=FakeCache(previous + tokens),
        )


class HuggingFaceAdapterTests(unittest.TestCase):
    def test_legacy_tuple_cache_crops_sequence_dimension(self) -> None:
        legacy = ((FakeLegacyKV(4), FakeLegacyKV(4)),)

        cropped = HuggingFaceCausalLM._crop_legacy_cache(legacy, 3)

        self.assertEqual(cropped[0][0].sequence_length, 3)
        self.assertEqual(cropped[0][1].sequence_length, 3)

    def test_adapter_scores_proposal_in_one_forward_pass(self) -> None:
        model = FakeModel()
        adapter = HuggingFaceCausalLM(model, FakeTokenizer(), FakeTorch())

        rows = adapter.score_proposal([0, 1], [0, 1])

        self.assertTrue(model.eval_called)
        self.assertEqual(model.calls, [([0, 1, 0, 1], False)])
        self.assertEqual(rows, [[0.25, 0.75], [0.75, 0.25], [0.25, 0.75]])

    def test_adapter_uses_pinned_transfer_and_named_cuda_stream(self) -> None:
        model = FakeModel()
        adapter = HuggingFaceCausalLM(model, FakeTokenizer(), FakeTorch())
        runtime = FakeCudaRuntime()
        adapter.configure_cuda_runtime(runtime, stream_role="target")

        rows = adapter.next_token_probs([0, 1])

        self.assertEqual(rows, [0.25, 0.75])
        self.assertEqual(
            [name for name, _, _ in runtime.transfers],
            ["target.input_ids", "target.attention_mask"],
        )
        role, dependencies, label = runtime.submissions[0]
        self.assertEqual(role, "target")
        self.assertEqual(len(dependencies), 2)
        self.assertEqual(label, "specdecode.target.forward")

    def test_cache_reuses_prefill_and_scores_only_proposal_suffix(self) -> None:
        model = FakeCachedModel()
        adapter = HuggingFaceCausalLM(
            model,
            FakeTokenizer(),
            FakeTorch(),
            use_kv_cache=True,
        )

        first = adapter.next_token_probs([0, 1])
        rows = adapter.score_proposal([0, 1], [0, 1])

        self.assertEqual(first, [0.2, 0.8])
        self.assertEqual(
            rows,
            [[0.2, 0.8], [0.3, 0.7], [0.4, 0.6]],
        )
        self.assertEqual(
            model.calls,
            [
                {
                    "input_tokens": [0, 1],
                    "attention_length": 2,
                    "past_length": 0,
                    "use_cache": True,
                },
                {
                    "input_tokens": [0, 1],
                    "attention_length": 4,
                    "past_length": 2,
                    "use_cache": True,
                },
            ],
        )
        self.assertEqual(adapter.cached_token_count, 4)
        self.assertEqual(adapter.cache_stats.full_prefills, 1)
        self.assertEqual(adapter.cache_stats.incremental_forwards, 1)
        self.assertEqual(adapter.cache_stats.exact_cache_hits, 1)

    def test_cache_crops_rejected_suffix_and_replays_correction(self) -> None:
        model = FakeCachedModel()
        adapter = HuggingFaceCausalLM(
            model,
            FakeTokenizer(),
            FakeTorch(),
            use_kv_cache=True,
        )
        adapter.score_proposal([0, 1], [0, 1])
        old_cache = adapter._past_key_values

        rows = adapter.score_proposal([0, 1, 0, 0], [1])

        self.assertEqual(rows, [[0.4, 0.6], [0.5, 0.5]])
        self.assertEqual(old_cache.crop_calls, [-1])
        self.assertEqual(model.calls[-1]["input_tokens"], [0, 1])
        self.assertEqual(model.calls[-1]["past_length"], 3)
        self.assertEqual(adapter.cached_token_count, 5)
        self.assertEqual(adapter.cache_stats.cropped_tokens, 1)

    def test_cache_reset_forces_a_new_prefill(self) -> None:
        model = FakeCachedModel()
        adapter = HuggingFaceCausalLM(
            model,
            FakeTokenizer(),
            FakeTorch(),
            use_kv_cache=True,
        )
        adapter.next_token_probs([0, 1])

        adapter.reset_cache()
        adapter.next_token_probs([1, 0])

        self.assertEqual([call["past_length"] for call in model.calls], [0, 0])
        self.assertEqual(adapter.cache_stats.full_prefills, 2)

    def test_cache_refills_when_rollback_exceeds_retained_window(self) -> None:
        model = FakeCachedModel()
        adapter = HuggingFaceCausalLM(
            model,
            FakeTokenizer(),
            FakeTorch(),
            use_kv_cache=True,
        )
        adapter._cache_tokens = (0, 1, 0, 1, 0, 1)
        adapter._past_key_values = FakeWindowCache([0, 1])
        adapter._cached_next_probabilities = (0.6, 0.4)

        probabilities = adapter.next_token_probs([0, 0])

        self.assertEqual(probabilities, [0.2, 0.8])
        self.assertEqual(model.calls[-1]["input_tokens"], [0, 0])
        self.assertEqual(model.calls[-1]["past_length"], 0)
        self.assertEqual(adapter.cache_stats.cropped_tokens, 6)

    def test_paged_mirror_tracks_model_updates_rollback_and_reset(self) -> None:
        model = FakeCachedModel()
        mirror = FakePagedCacheMirror()
        adapter = HuggingFaceCausalLM(
            model,
            FakeTokenizer(),
            FakeTorch(),
            use_kv_cache=True,
            paged_cache_mirror=mirror,
        )
        adapter.score_proposal([0, 1], [0, 1])

        adapter.score_proposal([0, 1, 0, 0], [1])
        adapter.reset_cache()

        self.assertEqual(len(mirror.synchronized), 2)
        self.assertEqual(mirror.truncated, [3])
        self.assertEqual(mirror.resets, 1)

    def test_pair_rejects_tokenizers_before_loading_model_weights(self) -> None:
        backends = (FakeTorch(), object(), FakeAutoTokenizer, "4.44.2")
        with mock.patch("specdecode.huggingface._require_backends", return_value=backends):
            with mock.patch.object(HuggingFaceCausalLM, "from_pretrained") as load_model:
                with self.assertRaises(TokenizerCompatibilityError):
                    HuggingFaceModelPair.from_pretrained("draft", "target")
        load_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
