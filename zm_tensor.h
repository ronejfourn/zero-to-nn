#pragma once

#include "zm_types.h"

struct zm_tensor;

#define ZM_TENSOR_BACKWARD_FXN(F) \
    void F (struct zm_tensor *this)

typedef ZM_TENSOR_BACKWARD_FXN((*zm_tensor_backward_fxn));

typedef struct zm_tensor {
    u32 dim;
    u32 *shape;
    f32 *data;

    f32 *grad;
    void *prev;
    u32 n_prev;
    void *backward_data;
    zm_tensor_backward_fxn backward;

    u32 *_offs;
    u32 _size_flat;
} zm_tensor;

#define zm_tensor_create(...) _zm_tensor_create(__VA_ARGS__, __FILE__, __LINE__)
#define zm_tensor_destroy(...) _zm_tensor_destroy(__VA_ARGS__, __FILE__, __LINE__)
zm_tensor _zm_tensor_create(u32 _dim, u32 *_shape, void *_data, bool require_grad, const char *file, u32 line);
void _zm_tensor_destroy(zm_tensor t, const char *file, u32 line);

#define zm_tensor_fill(...) _zm_tensor_fill(__VA_ARGS__, __FILE__, __LINE__)
#define zm_tensor_rand(...) _zm_tensor_rand(__VA_ARGS__, __FILE__, __LINE__)
#define zm_tensor_randn(...) _zm_tensor_randn(__VA_ARGS__, __FILE__, __LINE__)
zm_tensor _zm_tensor_fill(u32 _dim, u32 *_shape, f32 val, char *file, u32 line);
zm_tensor _zm_tensor_rand(u32 _dim, u32 *_shape, char *file, u32 line);
zm_tensor _zm_tensor_randn(u32 _dim, u32 *_shape, char *file, u32 line);
#define zm_tensor_zeros(...) _zm_tensor_fill(__VA_ARGS__, 0, __FILE__, __LINE__)
#define zm_tensor_ones(...) _zm_tensor_fill(__VA_ARGS__, 1, __FILE__, __LINE__)

#define zm_tensor_require_grad(...) _zm_tensor_require_grad(__VA_ARGS__, __FILE__, __LINE__)
void _zm_tensor_require_grad(zm_tensor *t, char *file, u32 line);

void zm_tensor_backward(zm_tensor *this);

void zm_tensor_print_data(const zm_tensor *t);
void zm_tensor_print_grad(const zm_tensor *t);

#define zm_tensor_set_prev(...) _zm_tensor_set_prev(__VA_ARGS__, __FILE__, __LINE__)
void _zm_tensor_set_prev(zm_tensor *t, void *p, u32 n, char *file, u32 line);
