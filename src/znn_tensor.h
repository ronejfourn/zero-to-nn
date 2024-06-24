#pragma once

#include "znn_types.h"

struct znn_tensor;

#define ZNN_TENSOR_BACKWARD_FXN(F) \
    void F (struct znn_tensor *this)

typedef ZNN_TENSOR_BACKWARD_FXN((*znn_tensor_backward_fxn));

#define ZNN_TENSOR_VIEW \
    bool is_view;      \
    u32 dim;           \
    u32 *shape;        \
    f32 *data;         \
    f32 *grad;         \
    u32 *step;         \
    u32 size

typedef struct znn_tensor_view {
    ZNN_TENSOR_VIEW;
} znn_tensor_view;

typedef struct znn_tensor {
    ZNN_TENSOR_VIEW;
    void *data_base;
    void *grad_base;
    void *prev;
    u32 n_prev;
    void *backward_data;
    znn_tensor_backward_fxn backward;
} znn_tensor;

#undef ZNN_TENSOR_VIEW

znn_tensor_view znn_tensor_as_tensor_view(znn_tensor *t);
znn_tensor znn_tensor_view_as_tensor(znn_tensor_view *v);
znn_tensor znn_tensor_from_view(znn_tensor_view *v);

#define znn_tensor_get_view(T, ...) _znn_tensor_get_view((znn_tensor_view *)T, __VA_ARGS__, -1)
znn_tensor_view _znn_tensor_get_view(znn_tensor_view *t, ...);
#define znn_tensor_get_range(T, A, B) _znn_tensor_get_range((znn_tensor_view *)T, A, B)
znn_tensor_view _znn_tensor_get_range(znn_tensor_view *t, u32 a, u32 b);

#define znn_tensor_create(...) _znn_tensor_create(__FILE__, __LINE__, __VA_ARGS__, -1)
znn_tensor _znn_tensor_create(const char *file, u32 line, ...);
#define znn_tensor_create_from_data(D, ...) _znn_tensor_create_from_data(__FILE__, __LINE__, D, __VA_ARGS__, -1)
znn_tensor _znn_tensor_create_from_data(const char *file, u32 line, void *data, ...);
#define znn_tensor_create_from_shape(D, S) _znn_tensor_create_from_shape(__FILE__, __LINE__, D, S)
znn_tensor _znn_tensor_create_from_shape(const char *file, u32 line, u32 dim, u32 *shape);
#define znn_tensor_destroy(T) _znn_tensor_destroy(__FILE__, __LINE__, T)
void _znn_tensor_destroy(const char *file, u32 line, znn_tensor t);

#define znn_tensor_fill(V, ...) _znn_tensor_fill(V, __FILE__, __LINE__, __VA_ARGS__, -1)
znn_tensor _znn_tensor_fill(f32 val, const char *file, u32 line, ...);
#define znn_tensor_zeros(...) _znn_tensor_fill(0, __FILE__, __LINE__, __VA_ARGS__, -1)
#define znn_tensor_ones(...) _znn_tensor_fill(1, __FILE__, __LINE__, __VA_ARGS__, -1)

#define znn_tensor_rand(...) _znn_tensor_rand(__FILE__, __LINE__, __VA_ARGS__, -1)
znn_tensor _znn_tensor_rand(const char *file, u32 line, ...);
#define znn_tensor_randn(...) _znn_tensor_randn(__FILE__, __LINE__, __VA_ARGS__, -1)
znn_tensor _znn_tensor_randn(const char *file, u32 line, ...);
#define znn_tensor_randr(A, B, ...) _znn_tensor_randr(__FILE__, __LINE__, A, B, __VA_ARGS__, -1)
znn_tensor _znn_tensor_randr(const char *file, u32 line, f32 a, f32 b,...);

#define znn_tensor_require_grad(T) _znn_tensor_require_grad(__FILE__, __LINE__, T)
void _znn_tensor_require_grad(char* file, u32 line, znn_tensor *t);

void znn_tensor_backward(znn_tensor *this);

#define znn_tensor_print_data(T) _znn_tensor_print_data((const znn_tensor_view *)T)
void _znn_tensor_print_data(const znn_tensor_view *t);
#define znn_tensor_print_grad(T) _znn_tensor_print_grad((const znn_tensor_view *)T)
void _znn_tensor_print_grad(const znn_tensor_view *t);
#define znn_tensor_print_shape(T) _znn_tensor_print_grad((const znn_tensor_view *)T)
void _znn_tensor_print_shape(const znn_tensor_view *t);

#define znn_tensor_set_prev(T, ...) _znn_tensor_set_prev(__FILE__, __LINE__, T, __VA_ARGS__, NULL)
void _znn_tensor_set_prev(char *file, u32 line, znn_tensor *this, ...);
