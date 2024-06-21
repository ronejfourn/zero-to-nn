#include "zm_util.h"

#include <stdlib.h>
#include <string.h>

void *zm_copy(const void*src, u32 size) {
    void *d = malloc(size);
    memcpy(d, src, size);
    return d;
}
