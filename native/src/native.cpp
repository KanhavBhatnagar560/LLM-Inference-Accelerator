#include "specdecode/native.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

struct SumResult {
    sd_status_t status;
    long double sum;
};

SumResult validate_and_sum(const double* values, const size_t count) noexcept {
    if (values == nullptr) {
        return {SD_STATUS_NULL_ARGUMENT, 0.0L};
    }
    if (count == 0) {
        return {SD_STATUS_INVALID_SIZE, 0.0L};
    }

    long double sum = 0.0L;
    long double correction = 0.0L;
    for (size_t index = 0; index < count; ++index) {
        const double value = values[index];
        if (!std::isfinite(value)) {
            return {SD_STATUS_NONFINITE_VALUE, 0.0L};
        }
        if (value < 0.0) {
            return {SD_STATUS_NEGATIVE_PROBABILITY, 0.0L};
        }

        const long double candidate = static_cast<long double>(value) - correction;
        const long double updated = sum + candidate;
        correction = (updated - sum) - candidate;
        sum = updated;
    }
    if (!(sum > 0.0L)) {
        return {SD_STATUS_ZERO_MASS, 0.0L};
    }
    return {SD_STATUS_OK, sum};
}

bool invalid_uniform(const double uniform) noexcept {
    return !std::isfinite(uniform) || uniform < 0.0 || uniform >= 1.0;
}

}  // namespace

extern "C" {

uint32_t sd_abi_version(void) {
    return SD_NATIVE_ABI_VERSION;
}

const char* sd_status_string(const sd_status_t status) {
    switch (status) {
        case SD_STATUS_OK:
            return "ok";
        case SD_STATUS_NULL_ARGUMENT:
            return "null argument";
        case SD_STATUS_INVALID_SIZE:
            return "invalid size";
        case SD_STATUS_NONFINITE_VALUE:
            return "non-finite value";
        case SD_STATUS_NEGATIVE_PROBABILITY:
            return "negative probability";
        case SD_STATUS_ZERO_MASS:
            return "zero probability mass";
        case SD_STATUS_TOKEN_OUT_OF_RANGE:
            return "token out of range";
        case SD_STATUS_INVALID_UNIFORM:
            return "uniform must be finite and in [0, 1)";
        case SD_STATUS_OVERFLOW:
            return "size overflow";
        case SD_STATUS_INTERNAL_ERROR:
            return "internal error";
        case SD_STATUS_INVALID_SCALE:
            return "quantization scale must be finite and non-negative";
    }
    return "unknown status";
}

sd_status_t sd_normalize_f64(const double* weights, const size_t count, double* output) {
    try {
        if (output == nullptr) {
            return SD_STATUS_NULL_ARGUMENT;
        }
        const SumResult result = validate_and_sum(weights, count);
        if (result.status != SD_STATUS_OK) {
            return result.status;
        }
        for (size_t index = 0; index < count; ++index) {
            output[index] = static_cast<double>(
                static_cast<long double>(weights[index]) / result.sum
            );
        }
        return SD_STATUS_OK;
    } catch (...) {
        return SD_STATUS_INTERNAL_ERROR;
    }
}

sd_status_t sd_sample_categorical_f64(
    const double* weights,
    const size_t count,
    const double uniform_01,
    uint64_t* output_token
) {
    try {
        if (output_token == nullptr) {
            return SD_STATUS_NULL_ARGUMENT;
        }
        if (invalid_uniform(uniform_01)) {
            return SD_STATUS_INVALID_UNIFORM;
        }
        const SumResult result = validate_and_sum(weights, count);
        if (result.status != SD_STATUS_OK) {
            return result.status;
        }

        const long double threshold = static_cast<long double>(uniform_01) * result.sum;
        long double cumulative = 0.0L;
        for (size_t index = 0; index < count; ++index) {
            cumulative += static_cast<long double>(weights[index]);
            if (threshold < cumulative) {
                *output_token = static_cast<uint64_t>(index);
                return SD_STATUS_OK;
            }
        }
        *output_token = static_cast<uint64_t>(count - 1);
        return SD_STATUS_OK;
    } catch (...) {
        return SD_STATUS_INTERNAL_ERROR;
    }
}

sd_status_t sd_residual_weights_f64(
    const double* target,
    const double* draft,
    const size_t vocabulary_size,
    double* output_weights
) {
    try {
        if (output_weights == nullptr) {
            return SD_STATUS_NULL_ARGUMENT;
        }
        const SumResult target_sum = validate_and_sum(target, vocabulary_size);
        if (target_sum.status != SD_STATUS_OK) {
            return target_sum.status;
        }
        const SumResult draft_sum = validate_and_sum(draft, vocabulary_size);
        if (draft_sum.status != SD_STATUS_OK) {
            return draft_sum.status;
        }

        long double residual_sum = 0.0L;
        for (size_t index = 0; index < vocabulary_size; ++index) {
            const long double p = static_cast<long double>(target[index]) / target_sum.sum;
            const long double q = static_cast<long double>(draft[index]) / draft_sum.sum;
            const double residual = static_cast<double>(std::max(0.0L, p - q));
            output_weights[index] = residual;
            residual_sum += static_cast<long double>(residual);
        }

        if (residual_sum <= 1.0e-15L) {
            for (size_t index = 0; index < vocabulary_size; ++index) {
                output_weights[index] = static_cast<double>(
                    static_cast<long double>(target[index]) / target_sum.sum
                );
            }
        }
        return SD_STATUS_OK;
    } catch (...) {
        return SD_STATUS_INTERNAL_ERROR;
    }
}

sd_status_t sd_acceptance_probabilities_f64(
    const double* target_rows,
    const double* draft_rows,
    const uint64_t* proposed_tokens,
    const size_t proposal_count,
    const size_t vocabulary_size,
    double* output_probabilities
) {
    try {
        if (
            target_rows == nullptr || draft_rows == nullptr || proposed_tokens == nullptr ||
            output_probabilities == nullptr
        ) {
            return SD_STATUS_NULL_ARGUMENT;
        }
        if (proposal_count == 0 || vocabulary_size == 0) {
            return SD_STATUS_INVALID_SIZE;
        }
        if (proposal_count > std::numeric_limits<size_t>::max() / vocabulary_size) {
            return SD_STATUS_OVERFLOW;
        }

        for (size_t row = 0; row < proposal_count; ++row) {
            if (proposed_tokens[row] >= vocabulary_size) {
                return SD_STATUS_TOKEN_OUT_OF_RANGE;
            }
            const size_t offset = row * vocabulary_size;
            const SumResult target_sum = validate_and_sum(target_rows + offset, vocabulary_size);
            if (target_sum.status != SD_STATUS_OK) {
                return target_sum.status;
            }
            const SumResult draft_sum = validate_and_sum(draft_rows + offset, vocabulary_size);
            if (draft_sum.status != SD_STATUS_OK) {
                return draft_sum.status;
            }
        }

        for (size_t row = 0; row < proposal_count; ++row) {
            const size_t offset = row * vocabulary_size;
            const SumResult target_sum = validate_and_sum(target_rows + offset, vocabulary_size);
            const SumResult draft_sum = validate_and_sum(draft_rows + offset, vocabulary_size);
            const size_t token = static_cast<size_t>(proposed_tokens[row]);
            const long double p =
                static_cast<long double>(target_rows[offset + token]) / target_sum.sum;
            const long double q =
                static_cast<long double>(draft_rows[offset + token]) / draft_sum.sum;
            output_probabilities[row] =
                q == 0.0L ? 1.0 : static_cast<double>(std::min(1.0L, p / q));
        }
        return SD_STATUS_OK;
    } catch (...) {
        return SD_STATUS_INTERNAL_ERROR;
    }
}

sd_status_t sd_first_rejection_f64(
    const double* acceptance_probabilities,
    const double* uniforms,
    const size_t proposal_count,
    size_t* output_accepted_count,
    size_t* output_rejection_index
) {
    try {
        if (output_accepted_count == nullptr || output_rejection_index == nullptr) {
            return SD_STATUS_NULL_ARGUMENT;
        }
        if (proposal_count > 0 && (acceptance_probabilities == nullptr || uniforms == nullptr)) {
            return SD_STATUS_NULL_ARGUMENT;
        }
        for (size_t index = 0; index < proposal_count; ++index) {
            const double probability = acceptance_probabilities[index];
            if (!std::isfinite(probability)) {
                return SD_STATUS_NONFINITE_VALUE;
            }
            if (probability < 0.0 || probability > 1.0) {
                return SD_STATUS_NEGATIVE_PROBABILITY;
            }
            if (invalid_uniform(uniforms[index])) {
                return SD_STATUS_INVALID_UNIFORM;
            }
        }
        for (size_t index = 0; index < proposal_count; ++index) {
            if (!(uniforms[index] < acceptance_probabilities[index])) {
                *output_accepted_count = index;
                *output_rejection_index = index;
                return SD_STATUS_OK;
            }
        }
        *output_accepted_count = proposal_count;
        *output_rejection_index = proposal_count;
        return SD_STATUS_OK;
    } catch (...) {
        return SD_STATUS_INTERNAL_ERROR;
    }
}

sd_status_t sd_quantize_symmetric_int8_f64(
    const double* values,
    const size_t count,
    int8_t* output_values,
    double* output_scale
) {
    try {
        if (values == nullptr || output_values == nullptr || output_scale == nullptr) {
            return SD_STATUS_NULL_ARGUMENT;
        }
        if (count == 0) {
            return SD_STATUS_INVALID_SIZE;
        }

        double maximum = 0.0;
        for (size_t index = 0; index < count; ++index) {
            if (!std::isfinite(values[index])) {
                return SD_STATUS_NONFINITE_VALUE;
            }
            maximum = std::max(maximum, std::abs(values[index]));
        }

        if (maximum == 0.0) {
            for (size_t index = 0; index < count; ++index) {
                output_values[index] = INT8_C(0);
            }
            *output_scale = 0.0;
            return SD_STATUS_OK;
        }

        const float float_scale = static_cast<float>(maximum / 127.0);
        if (!std::isfinite(float_scale) || float_scale == 0.0F) {
            return SD_STATUS_INVALID_SCALE;
        }
        const double scale = static_cast<double>(float_scale);
        for (size_t index = 0; index < count; ++index) {
            const double rounded = std::round(values[index] / scale);
            const double clamped = std::max(-127.0, std::min(127.0, rounded));
            output_values[index] = static_cast<int8_t>(clamped);
        }
        *output_scale = scale;
        return SD_STATUS_OK;
    } catch (...) {
        return SD_STATUS_INTERNAL_ERROR;
    }
}

sd_status_t sd_dequantize_symmetric_int8_f64(
    const int8_t* values,
    const size_t count,
    const double scale,
    double* output_values
) {
    try {
        if (values == nullptr || output_values == nullptr) {
            return SD_STATUS_NULL_ARGUMENT;
        }
        if (count == 0) {
            return SD_STATUS_INVALID_SIZE;
        }
        if (!std::isfinite(scale) || scale < 0.0) {
            return SD_STATUS_INVALID_SCALE;
        }
        if (scale == 0.0) {
            for (size_t index = 0; index < count; ++index) {
                if (values[index] != INT8_C(0)) {
                    return SD_STATUS_INVALID_SCALE;
                }
            }
        }

        for (size_t index = 0; index < count; ++index) {
            output_values[index] = static_cast<double>(values[index]) * scale;
        }
        return SD_STATUS_OK;
    } catch (...) {
        return SD_STATUS_INTERNAL_ERROR;
    }
}

}  // extern "C"
