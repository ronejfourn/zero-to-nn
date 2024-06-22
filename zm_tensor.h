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
    u32 *step;
    u32 size;

    void *prev;
    u32 n_prev;
    void *backward_data;
    zm_tensor_backward_fxn backward;
} zm_tensor;

#define zm_tensor_create(...) _zm_tensor_create(__FILE__, __LINE__, __VA_ARGS__, -1)
zm_tensor _zm_tensor_create(const char *file, u32 line, ...);
#define zm_tensor_create_from_shape(D, S) _zm_tensor_create_from_shape(__FILE__, __LINE__, D, S)
zm_tensor _zm_tensor_create_from_shape(const char *file, u32 line, u32 dim, u32 *shape);
#define zm_tensor_create_from_data(D, S, P) _zm_tensor_create_from_data(__FILE__, __LINE__, D, S, P)
zm_tensor _zm_tensor_create_from_data(const char *file, u32 line, u32 dim, u32 *shape, void *data);
#define zm_tensor_destroy(T) _zm_tensor_destroy(__FILE__, __LINE__, T)
void _zm_tensor_destroy(const char *file, u32 line, zm_tensor t);

#define zm_tensor_fill(V, ...) _zm_tensor_fill(__FILE__, __LINE__, V, __VA_ARGS__, -1)
zm_tensor _zm_tensor_fill(const char *file, u32 line, f32 val, ...);
#define zm_tensor_zeros(...) _zm_tensor_fill(__FILE__, __LINE__, 0, __VA_ARGS__, -1)
#define zm_tensor_ones(...) _zm_tensor_fill(__FILE__, __LINE__, 1, __VA_ARGS__, -1)

#define zm_tensor_rand(...) _zm_tensor_rand(__FILE__, __LINE__, __VA_ARGS__, -1)
zm_tensor _zm_tensor_rand(const char *file, u32 line, ...);
#define zm_tensor_randn(...) _zm_tensor_randn(__FILE__, __LINE__, __VA_ARGS__, -1)
zm_tensor _zm_tensor_randn(const char *file, u32 line, ...);

#define zm_tensor_require_grad(T) _zm_tensor_require_grad(__FILE__, __LINE__, T)
void _zm_tensor_require_grad(char* file, u32 line, zm_tensor *t);

void zm_tensor_backward(zm_tensor *this);

void zm_tensor_print_data(const zm_tensor *t);
void zm_tensor_print_grad(const zm_tensor *t);

#define zm_tensor_set_prev(T, ...) _zm_tensor_set_prev(__FILE__, __LINE__, T, __VA_ARGS__, NULL)
void _zm_tensor_set_prev(char *file, u32 line, zm_tensor *this, ...);
