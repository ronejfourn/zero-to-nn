#include "zm_util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void *_zm_copy(const void*src, u32 size, const char *file, u32 line) {
    void *d = _zm_malloc(size, file, line);
    if (src) memcpy(d, src, size);
    return d;
}

void *_zm_malloc(u32 size, const char *file, u32 line) {
    void *d = malloc(size);
#ifdef ZM_TRACE_ENABLE
    printf(ZM_TRACE_FMT"(%d) -> %p\n", file, line, "malloc", size, d);
#endif
    return d;
}

void _zm_free(void *ptr, const char *file, u32 line) {
    free(ptr);
#ifdef ZM_TRACE_ENABLE
    printf(ZM_TRACE_FMT"(%p)\n", file, line, "free", ptr);
#endif
}
