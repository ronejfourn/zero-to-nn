#pragma once

#include "znn_common.h"

struct znn_tensor;

#define ZNN_TENSOR_BACKWARD_FXN(F) \
    void F (struct znn_tensor *this)

typedef ZNN_TENSOR_BACKWARD_FXN((*znn_tensor_backward_fxn));

#define ZNN_TENSOR_VIEW \
    u32 dim;            \
    u32 *shape;         \
    f32 *data;          \
    f32 *grad;          \
    u32 size

typedef struct znn_tensor_view {
    ZNN_TENSOR_VIEW;
} znn_tensor_view;

typedef struct znn_tensor {
    ZNN_TENSOR_VIEW;
    void *prev;
    u32 n_prev;
    void *backward_data;
    ZNN_FXN(znn_tensor_backward_fxn) backward;
} znn_tensor;

#undef ZNN_TENSOR_VIEW

#define znn_tensor_divide(T, S) _znn_tensor_divide((znn_tensor_view *)(T), S)
void _znn_tensor_divide(znn_tensor_view *t, f32 s);

#define znn_tensor_one_hot(I, N) _znn_tensor_one_hot(__FILE__, __LINE__, (znn_tensor_view *)(I), N)
znn_tensor _znn_tensor_one_hot(const char *file, u32 line, znn_tensor_view *i, u32 num_classes);

#define znn_tensor_create(...) _znn_tensor_create(__FILE__, __LINE__, __VA_ARGS__, -1)
znn_tensor _znn_tensor_create(const char *file, u32 line, ...);
#define znn_tensor_from_data(D, ...) _znn_tensor_from_data(__FILE__, __LINE__, D, __VA_ARGS__, -1)
znn_tensor _znn_tensor_from_data(const char *file, u32 line, void *data, ...);
#define znn_tensor_from_shape(D, S) _znn_tensor_from_shape(__FILE__, __LINE__, D, S)
znn_tensor _znn_tensor_from_shape(const char *file, u32 line, u32 dim, u32 *shape);
#define znn_tensor_copy(V) _znn_tensor_from_view(__FILE__, __LINE__, (znn_tensor_from_view *)(V))
znn_tensor _znn_tensor_copy(const char *file, u32 line, const znn_tensor_view *v);

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

#define znn_tensor_destroy(T) _znn_tensor_destroy(__FILE__, __LINE__, T)
void _znn_tensor_destroy(const char *file, u32 line, znn_tensor t);

#define znn_tensor_get(T, V, ...) _znn_tensor_get((const znn_tensor_view *)(T), V, ##__VA_ARGS__, -1)
void _znn_tensor_get(const znn_tensor_view *t, znn_tensor_view *v, ...);

#define znn_tensor_require_grad(T) _znn_tensor_require_grad(__FILE__, __LINE__, T)
void _znn_tensor_require_grad(char* file, u32 line, znn_tensor *t);

void znn_tensor_backward(znn_tensor *this);

typedef enum {
    ZNN_TENSOR_PRINT_DATA  = 1 << 0,
    ZNN_TENSOR_PRINT_GRAD  = 1 << 1,
    ZNN_TENSOR_PRINT_SHAPE = 1 << 2,
} znn_tensor_print_opts;

#define znn_tensor_print(V, O) _znn_tensor_print((const znn_tensor_view *)(V), O)
void _znn_tensor_print(const znn_tensor_view *v, u32 opts);

#define znn_tensor_set_prev(T, ...) _znn_tensor_set_prev(__FILE__, __LINE__, T, __VA_ARGS__, NULL)
void _znn_tensor_set_prev(char *file, u32 line, znn_tensor *this, ...);
