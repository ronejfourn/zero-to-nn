#include "zm_util.h"
#include <stdlib.h>

void *_zm_copy(const void*src, u32 size, const char *file, u32 line) {
    void *d = _zm_malloc(size, file, line);
    if (src) memcpy(d, src, size);
    return d;
}

void *_zm_malloc(u32 size, const char *file, u32 line) {
    void *d = calloc(size, 1);
#if ZM_TRACE_ENABLE
    printf(ZM_TRACE_FMT"(%d) -> %p\n", file, line, "malloc", size, d);
#endif
    return d;
}

void *zm_alignptr16(void *p) {
	return (void *)(((uintptr_t)p & ~15) + 16);
}

void _zm_free(void *ptr, const char *file, u32 line) {
    free(ptr);
#if ZM_TRACE_ENABLE
    printf(ZM_TRACE_FMT"(%p)\n", file, line, "free", ptr);
#endif
}
