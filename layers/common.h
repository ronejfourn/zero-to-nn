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

static inline void _update(zm_layer *this, zm_tensor *input, zm_tensor_backward_fxn fn, zm_tensor n) {
    this->input = input;
    zm_tensor_destroy(this->output);
    this->output = n;
    zm_tensor_require_grad(&this->output);
    this->output.backward = fn;
}
