#pragma once

#include "zm_types.h"

void *zm_copy(const void*src, u32 size);

#define zm_arraylen(_A) (sizeof(_A) / sizeof(*(_A)))
