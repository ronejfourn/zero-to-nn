#pragma once
#include "../zm_util.h"
#include "../zm_layers.h"
#include <math.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>

#define L(_L) \
    zm_layer l = {0}; \
    l.forward = zm_layer_forward_ ## _L; \
    l.destroy = zm_layer_destroy_ ## _L

static inline void _set_prev(zm_tensor *t, zm_tensor **p, u32 n) {
    t->prev = zm_copy(p, sizeof(*p) * n);
    t->n_prev = n;
}
