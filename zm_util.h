#pragma once

#include "zm_types.h"
#include <stdio.h>

#if ZM_TRACE_ENABLE
#define ZM_TRACE_FMT "%s:%d: %s"
#define zm_trace(_FILE, _LINE) printf(ZM_TRACE_FMT"\n", _FILE, _LINE, __FUNCTION__)
#else
#define zm_trace(...)
#endif

#define zm_arraylen(_A) (sizeof(_A) / sizeof(*(_A)))

#define zm_malloc(_SIZE) _zm_malloc(_SIZE, __FILE__, __LINE__)
void *_zm_malloc(u32 size, const char *file, u32 line);

#define zm_free(_PTR) _zm_free(_PTR, __FILE__, __LINE__)
void _zm_free(void *ptr, const char *file, u32 line);

#define zm_copy(_SRC, _SIZE) _zm_copy(_SRC, _SIZE, __FILE__, __LINE__)
void *_zm_copy(const void*src, u32 size, const char *file, u32 line);
