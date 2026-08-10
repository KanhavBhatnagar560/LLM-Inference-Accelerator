import unittest
from types import SimpleNamespace
from unittest import mock

from specdecode.huggingface import HuggingFaceCausalLM, HuggingFaceModelPair
from specdecode.tokenizers import TokenizerCompatibilityError


class FakeInput:
    def __init__(self, values):
        self.values = values


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


class HuggingFaceAdapterTests(unittest.TestCase):
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

    def test_pair_rejects_tokenizers_before_loading_model_weights(self) -> None:
        backends = (FakeTorch(), object(), FakeAutoTokenizer, "4.44.2")
        with mock.patch("specdecode.huggingface._require_backends", return_value=backends):
            with mock.patch.object(HuggingFaceCausalLM, "from_pretrained") as load_model:
                with self.assertRaises(TokenizerCompatibilityError):
                    HuggingFaceModelPair.from_pretrained("draft", "target")
        load_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
