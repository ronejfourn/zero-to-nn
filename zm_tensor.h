#pragma once

#include "zm_types.h"

typedef struct zm_tensor {
    u32 dim;
    u32 *shape;
    f32 *data;

    f32 *grad;
    // grad_func

    struct zm_tensor **prev;
    u32 prev_count;

    u32 *_offs;
    u32 _size_flat;
} zm_tensor;

zm_tensor zm_tensor_create(u32 _dim, u32 *_shape, void *_data);
void zm_tensor_destroy(zm_tensor t);

zm_tensor zm_tensor_zeros(u32 _dim, u32 *_shape);
zm_tensor zm_tensor_ones(u32 _dim, u32 *_shape);
zm_tensor zm_tensor_random(u32 _dim, u32 *_shape);
zm_tensor zm_tensor_random_n(u32 _dim, u32 *_shape);
zm_tensor zm_tensor_random_r(u32 _dim, u32 *_shape, f32 a, f32 b);

void zm_tensor_require_grad(zm_tensor *t);

void zm_tensor_print(zm_tensor *t);
