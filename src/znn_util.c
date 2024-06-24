#include "znn_util.h"
#include <stdlib.h>

void *_znn_copy(const void*src, u32 size, const char *file, u32 line) {
    void *d = _znn_malloc(size, file, line);
    if (src) memcpy(d, src, size);
    return d;
}

void *_znn_malloc(u32 size, const char *file, u32 line) {
    void *d = calloc(size, 1);
#if znn_TRACE_ENABLE
    printf(znn_TRACE_FMT"(%d) -> %p\n", file, line, "malloc", size, d);
#endif
    return d;
}

void _znn_free(void *ptr, const char *file, u32 line) {
    free(ptr);
#if znn_TRACE_ENABLE
    printf(znn_TRACE_FMT"(%p)\n", file, line, "free", ptr);
#endif
}
