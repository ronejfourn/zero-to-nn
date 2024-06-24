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

u16 _znn_correct_endian_little16(u16 n) { return n << 8 | n >> 8; }
u16 _znn_correct_endian_big16(u16 n) { return n; }

u32 _znn_correct_endian_little32(u32 n) { return
    (n & 0x000000ff) << 24 |
    (n & 0x0000ff00) << 8  |
    (n & 0x00ff0000) >> 8  |
    (n & 0xff000000) >> 24 ; }
u32 _znn_correct_endian_big32(u32 n) { return n; }

u64 _znn_correct_endian_little64(u64 n) { return
    (n & 0x00000000000000ff) << 56 |
    (n & 0x000000000000ff00) << 40 |
    (n & 0x0000000000ff0000) << 24 |
    (n & 0x00000000ff000000) << 8  |
    (n & 0x000000ff00000000) >> 8  |
    (n & 0x0000ff0000000000) >> 24 |
    (n & 0x00ff000000000000) >> 40 |
    (n & 0xff00000000000000) >> 56 ; }
u64 _znn_correct_endian_big64(u64 n) { return n; }

void _znn_check_endian() {
    u8 b[4] = {1, 0, 0, 0};
    if (*(u32*)b == 1) {
        znn_correct_endian16 = _znn_correct_endian_little16;
        znn_correct_endian32 = _znn_correct_endian_little32;
        znn_correct_endian64 = _znn_correct_endian_little64;
    } else {
        znn_correct_endian16 = _znn_correct_endian_big16;
        znn_correct_endian32 = _znn_correct_endian_big32;
        znn_correct_endian64 = _znn_correct_endian_big64;
    }
}

u16 _znn_correct_endian_unknown16(u16 n) { _znn_check_endian(); return znn_correct_endian16(n); }
u32 _znn_correct_endian_unknown32(u32 n) { _znn_check_endian(); return znn_correct_endian32(n); }
u64 _znn_correct_endian_unknown64(u64 n) { _znn_check_endian(); return znn_correct_endian64(n); }
u16 (*znn_correct_endian16)(u16 n) = _znn_correct_endian_unknown16;
u32 (*znn_correct_endian32)(u32 n) = _znn_correct_endian_unknown32;
u64 (*znn_correct_endian64)(u64 n) = _znn_correct_endian_unknown64;
