#include "specdecode/native.h"

int main(void) {
    return sd_abi_version() == SD_NATIVE_ABI_VERSION ? 0 : 1;
}
