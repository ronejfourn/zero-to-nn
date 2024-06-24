#pragma once

#include "znn_types.h"
#include <stdio.h>
#include <assert.h>
#include <string.h>

#if ZNN_TRACE_ENABLE
#define ZNN_TRACE_FMT "%s:%d: %s"
#define znn_trace(_FILE, _LINE) printf(ZNN_TRACE_FMT"\n", _FILE, _LINE, __FUNCTION__)
#else
#define znn_trace(...)
#endif

#define znn_arraylen(_A) (sizeof(_A) / sizeof(*(_A)))

#define znn_malloc(_SIZE) _znn_malloc(_SIZE, __FILE__, __LINE__)
void *_znn_malloc(u32 size, const char *file, u32 line);

#define znn_free(_PTR) _znn_free(_PTR, __FILE__, __LINE__)
void _znn_free(void *ptr, const char *file, u32 line);

#define znn_copy(_SRC, _SIZE) _znn_copy(_SRC, _SIZE, __FILE__, __LINE__)
void *_znn_copy(const void*src, u32 size, const char *file, u32 line);
