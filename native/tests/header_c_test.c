#include "specdecode/native.h"

int main(void) {
    const double input[] = {0.0};
    int8_t output[] = {1};
    double scale = -1.0;
    if (sd_abi_version() != SD_NATIVE_ABI_VERSION) {
        return 1;
    }
    if (sd_quantize_symmetric_int8_f64(input, 1, output, &scale) != SD_STATUS_OK) {
        return 2;
    }
    return output[0] == 0 && scale == 0.0 ? 0 : 3;
}
