#include "specdecode/native.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {

bool close(const double left, const double right, const double tolerance = 1.0e-12) {
    return std::abs(left - right) <= tolerance;
}

}  // namespace

int main() {
    assert(sd_abi_version() == SD_NATIVE_ABI_VERSION);
    assert(sd_status_string(SD_STATUS_OK) != nullptr);

    double weights[] = {2.0, 3.0};
    assert(sd_normalize_f64(weights, 2, weights) == SD_STATUS_OK);
    assert(close(weights[0], 0.4));
    assert(close(weights[1], 0.6));

    const double invalid[] = {0.5, -0.5};
    double untouched[] = {7.0, 8.0};
    assert(sd_normalize_f64(invalid, 2, untouched) == SD_STATUS_NEGATIVE_PROBABILITY);
    assert(untouched[0] == 7.0 && untouched[1] == 8.0);

    const double zero[] = {0.0, 0.0};
    assert(sd_normalize_f64(zero, 2, untouched) == SD_STATUS_ZERO_MASS);
    const double nonfinite[] = {0.5, std::numeric_limits<double>::infinity()};
    assert(sd_normalize_f64(nonfinite, 2, untouched) == SD_STATUS_NONFINITE_VALUE);
    assert(sd_normalize_f64(nullptr, 2, untouched) == SD_STATUS_NULL_ARGUMENT);

    const double categorical[] = {2.0, 3.0};
    uint64_t token = 99;
    assert(sd_sample_categorical_f64(categorical, 2, 0.0, &token) == SD_STATUS_OK);
    assert(token == 0);
    assert(sd_sample_categorical_f64(categorical, 2, 0.4, &token) == SD_STATUS_OK);
    assert(token == 1);
    assert(sd_sample_categorical_f64(categorical, 2, 1.0, &token) == SD_STATUS_INVALID_UNIFORM);

    const double target[] = {0.25, 0.75};
    const double draft[] = {0.75, 0.25};
    double residual[] = {-1.0, -1.0};
    assert(sd_residual_weights_f64(target, draft, 2, residual) == SD_STATUS_OK);
    assert(close(residual[0], 0.0));
    assert(close(residual[1], 0.5));

    double fallback[] = {-1.0, -1.0};
    assert(sd_residual_weights_f64(target, target, 2, fallback) == SD_STATUS_OK);
    assert(close(fallback[0], 0.25));
    assert(close(fallback[1], 0.75));

    const double target_rows[] = {0.2, 0.5, 0.3, 0.2, 0.6, 0.2};
    const double draft_rows[] = {0.4, 0.4, 0.2, 0.6, 0.2, 0.2};
    const uint64_t proposals[] = {0, 1};
    double acceptance[] = {-1.0, -1.0};
    assert(
        sd_acceptance_probabilities_f64(
            target_rows, draft_rows, proposals, 2, 3, acceptance
        ) == SD_STATUS_OK
    );
    assert(close(acceptance[0], 0.5));
    assert(close(acceptance[1], 1.0));

    const double q_zero_target[] = {0.0, 1.0};
    const double q_zero_draft[] = {0.0, 1.0};
    const uint64_t q_zero_token[] = {0};
    double q_zero_acceptance[] = {-1.0};
    assert(
        sd_acceptance_probabilities_f64(
            q_zero_target, q_zero_draft, q_zero_token, 1, 2, q_zero_acceptance
        ) == SD_STATUS_OK
    );
    assert(q_zero_acceptance[0] == 1.0);

    const double uniforms_reject[] = {0.5, 0.1};
    size_t accepted = 99;
    size_t rejected = 99;
    assert(
        sd_first_rejection_f64(acceptance, uniforms_reject, 2, &accepted, &rejected) ==
        SD_STATUS_OK
    );
    assert(accepted == 0 && rejected == 0);

    const double uniforms_accept[] = {0.49, 0.1};
    assert(
        sd_first_rejection_f64(acceptance, uniforms_accept, 2, &accepted, &rejected) ==
        SD_STATUS_OK
    );
    assert(accepted == 2 && rejected == 2);

    const double middle_uniforms[] = {0.49, 1.0 - 1.0e-12};
    const double middle_acceptance[] = {0.5, 0.5};
    assert(
        sd_first_rejection_f64(
            middle_acceptance, middle_uniforms, 2, &accepted, &rejected
        ) == SD_STATUS_OK
    );
    assert(accepted == 1 && rejected == 1);

    const double zero_acceptance[] = {0.0};
    const double zero_uniform[] = {0.0};
    assert(
        sd_first_rejection_f64(
            zero_acceptance, zero_uniform, 1, &accepted, &rejected
        ) == SD_STATUS_OK
    );
    assert(accepted == 0 && rejected == 0);

    const uint64_t invalid_proposal[] = {3};
    double unchanged_acceptance[] = {9.0};
    assert(
        sd_acceptance_probabilities_f64(
            target_rows, draft_rows, invalid_proposal, 1, 3, unchanged_acceptance
        ) == SD_STATUS_TOKEN_OUT_OF_RANGE
    );
    assert(unchanged_acceptance[0] == 9.0);

    const double one[] = {1.0};
    const uint64_t token_zero[] = {0};
    assert(
        sd_acceptance_probabilities_f64(
            one,
            one,
            token_zero,
            std::numeric_limits<size_t>::max(),
            2,
            unchanged_acceptance
        ) == SD_STATUS_OVERFLOW
    );

    const double kv_values[] = {-1.0, -0.5, 0.0, 0.5, 1.0};
    int8_t quantized[] = {9, 9, 9, 9, 9};
    double scale = -1.0;
    assert(
        sd_quantize_symmetric_int8_f64(kv_values, 5, quantized, &scale) ==
        SD_STATUS_OK
    );
    assert(close(scale, static_cast<double>(static_cast<float>(1.0 / 127.0))));
    assert(quantized[0] == -127);
    assert(quantized[1] == -64);
    assert(quantized[2] == 0);
    assert(quantized[3] == 64);
    assert(quantized[4] == 127);

    double dequantized[] = {9.0, 9.0, 9.0, 9.0, 9.0};
    assert(
        sd_dequantize_symmetric_int8_f64(quantized, 5, scale, dequantized) ==
        SD_STATUS_OK
    );
    for (size_t index = 0; index < 5; ++index) {
        assert(std::abs(kv_values[index] - dequantized[index]) <= scale / 2.0 + 1.0e-15);
    }

    const double zero_kv[] = {0.0, 0.0};
    int8_t zero_quantized[] = {9, 9};
    assert(
        sd_quantize_symmetric_int8_f64(zero_kv, 2, zero_quantized, &scale) ==
        SD_STATUS_OK
    );
    assert(scale == 0.0);
    assert(zero_quantized[0] == 0 && zero_quantized[1] == 0);

    const double invalid_kv[] = {1.0, std::numeric_limits<double>::quiet_NaN()};
    int8_t untouched_quantized[] = {7, 8};
    scale = 9.0;
    assert(
        sd_quantize_symmetric_int8_f64(
            invalid_kv, 2, untouched_quantized, &scale
        ) == SD_STATUS_NONFINITE_VALUE
    );
    assert(untouched_quantized[0] == 7 && untouched_quantized[1] == 8);
    assert(scale == 9.0);

    const int8_t invalid_zero_scale[] = {1};
    double untouched_dequantized[] = {8.0};
    assert(
        sd_dequantize_symmetric_int8_f64(
            invalid_zero_scale, 1, 0.0, untouched_dequantized
        ) == SD_STATUS_INVALID_SCALE
    );
    assert(untouched_dequantized[0] == 8.0);
    assert(
        sd_dequantize_symmetric_int8_f64(
            invalid_zero_scale, 1, -1.0, untouched_dequantized
        ) == SD_STATUS_INVALID_SCALE
    );

    return 0;
}
