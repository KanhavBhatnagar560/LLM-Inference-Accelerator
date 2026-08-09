#ifndef SPECDECODE_NATIVE_H
#define SPECDECODE_NATIVE_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(SD_NATIVE_BUILD)
#define SD_API __declspec(dllexport)
#else
#define SD_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define SD_API __attribute__((visibility("default")))
#else
#define SD_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SD_NATIVE_ABI_VERSION UINT32_C(0x00010001)

typedef enum sd_status {
    SD_STATUS_OK = 0,
    SD_STATUS_NULL_ARGUMENT = 1,
    SD_STATUS_INVALID_SIZE = 2,
    SD_STATUS_NONFINITE_VALUE = 3,
    SD_STATUS_NEGATIVE_PROBABILITY = 4,
    SD_STATUS_ZERO_MASS = 5,
    SD_STATUS_TOKEN_OUT_OF_RANGE = 6,
    SD_STATUS_INVALID_UNIFORM = 7,
    SD_STATUS_OVERFLOW = 8,
    SD_STATUS_INTERNAL_ERROR = 9,
    SD_STATUS_INVALID_SCALE = 10
} sd_status_t;

SD_API uint32_t sd_abi_version(void);
SD_API const char* sd_status_string(sd_status_t status);

/* Normalize finite, non-negative weights. Input and output may alias. */
SD_API sd_status_t sd_normalize_f64(const double* weights, size_t count, double* output);

/* Sample from finite, non-negative weights using an explicit uniform in [0, 1). */
SD_API sd_status_t sd_sample_categorical_f64(
    const double* weights,
    size_t count,
    double uniform_01,
    uint64_t* output_token
);

/*
 * Build unnormalized max(0, normalize(target) - normalize(draft)) weights.
 * If residual mass is <= 1e-15, normalized target weights are returned.
 */
SD_API sd_status_t sd_residual_weights_f64(
    const double* target,
    const double* draft,
    size_t vocabulary_size,
    double* output_weights
);

/*
 * Compute one acceptance probability per row. Target and draft are flattened
 * row-major arrays with proposal_count * vocabulary_size entries.
 */
SD_API sd_status_t sd_acceptance_probabilities_f64(
    const double* target_rows,
    const double* draft_rows,
    const uint64_t* proposed_tokens,
    size_t proposal_count,
    size_t vocabulary_size,
    double* output_probabilities
);

/*
 * Find the first index where uniform < acceptance_probability is false.
 * Both outputs equal proposal_count when every proposal is accepted.
 */
SD_API sd_status_t sd_first_rejection_f64(
    const double* acceptance_probabilities,
    const double* uniforms,
    size_t proposal_count,
    size_t* output_accepted_count,
    size_t* output_rejection_index
);

/*
 * Symmetrically quantize one finite vector to [-127, 127]. The scale is
 * float32(max(abs(values)) / 127), or zero for an all-zero vector.
 */
SD_API sd_status_t sd_quantize_symmetric_int8_f64(
    const double* values,
    size_t count,
    int8_t* output_values,
    double* output_scale
);

/* Dequantize one symmetric INT8 vector with a finite, non-negative scale. */
SD_API sd_status_t sd_dequantize_symmetric_int8_f64(
    const int8_t* values,
    size_t count,
    double scale,
    double* output_values
);

#ifdef __cplusplus
}
#endif

#endif
