#pragma once

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>

typedef uint64_t u64;
typedef uint32_t u32;
typedef uint16_t u16;
typedef uint8_t  u8;

typedef int64_t  i64;
typedef int32_t  i32;
typedef int16_t  i16;
typedef int8_t   i8;

typedef float    f32;
typedef double   f64;

#ifndef __cplusplus
typedef enum { false, true } bool;
#endif

#if ZNN_TRACE_ENABLE
#define ZNN_TRACE_FMT "%s:%d: %s"
#define znn_trace(_FILE, _LINE) printf(ZNN_TRACE_FMT"\n", _FILE, _LINE, __FUNCTION__)
#else
#define znn_trace(...)
#endif

#define ZNN_FXN(F) struct {F fn; const char *name;}
#define ZNN_FXN_SET(V, F) do { (V).fn = F; (V).name = #F; } while (0)

#define znn_unimplemented() assert(!"implemented")
#define znn_unreachable() assert(!"reachable")

#define znn_arraylen(_A) (sizeof(_A) / sizeof(*(_A)))

u32 znn_randint();
f32 znn_rand();
f32 znn_randn();
f32 znn_randr(f32 a, f32 b);

#define znn_malloc(_SIZE) _znn_malloc(_SIZE, __FILE__, __LINE__)
void *_znn_malloc(u32 size, const char *file, u32 line);

#define znn_free(_PTR) _znn_free(_PTR, __FILE__, __LINE__)
void _znn_free(void *ptr, const char *file, u32 line);

#define znn_copy(_SRC, _SIZE) _znn_copy(_SRC, _SIZE, __FILE__, __LINE__)
void *_znn_copy(const void*src, u32 size, const char *file, u32 line);

extern u16 (*znn_correct_endian16)(u16 n);
extern u32 (*znn_correct_endian32)(u32 n);
extern u64 (*znn_correct_endian64)(u64 n);
