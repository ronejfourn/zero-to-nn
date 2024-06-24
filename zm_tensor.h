#pragma once

#include "zm_types.h"

struct zm_tensor;

#define ZM_TENSOR_BACKWARD_FXN(F) \
    void F (struct zm_tensor *this)

typedef ZM_TENSOR_BACKWARD_FXN((*zm_tensor_backward_fxn));

#define ZM_TENSOR_VIEW \
    bool is_view;      \
    u32 dim;           \
    u32 *shape;        \
    f32 *data;         \
    f32 *grad;         \
    u32 *step;         \
    u32 size

#pragma pack(push, 8)
typedef struct zm_tensor_view {
    ZM_TENSOR_VIEW;
} zm_tensor_view;

typedef struct zm_tensor {
    ZM_TENSOR_VIEW;
    void *data_base;
    void *grad_base;
    void *prev;
    u32 n_prev;
    void *backward_data;
    zm_tensor_backward_fxn backward;
} zm_tensor;
#pragma pack(pop)

#undef ZM_TENSOR_VIEW

zm_tensor_view zm_tensor_as_tensor_view(zm_tensor *t);
zm_tensor zm_tensor_view_as_tensor(zm_tensor_view *v);
zm_tensor zm_tensor_from_view(zm_tensor_view *v);

#define zm_tensor_get_view(T, ...) _zm_tensor_get_view((zm_tensor_view *)T, __VA_ARGS__, -1)
zm_tensor_view _zm_tensor_get_view(zm_tensor_view *t, ...);
#define zm_tensor_get_range(T, A, B) _zm_tensor_get_range((zm_tensor_view *)T, A, B)
zm_tensor_view _zm_tensor_get_range(zm_tensor_view *t, u32 a, u32 b);

#define zm_tensor_create(...) _zm_tensor_create(__FILE__, __LINE__, __VA_ARGS__, -1)
zm_tensor _zm_tensor_create(const char *file, u32 line, ...);
#define zm_tensor_create_from_data(D, ...) _zm_tensor_create_from_data(__FILE__, __LINE__, D, __VA_ARGS__, -1)
zm_tensor _zm_tensor_create_from_data(const char *file, u32 line, void *data, ...);
#define zm_tensor_create_from_shape(D, S) _zm_tensor_create_from_shape(__FILE__, __LINE__, D, S)
zm_tensor _zm_tensor_create_from_shape(const char *file, u32 line, u32 dim, u32 *shape);
#define zm_tensor_destroy(T) _zm_tensor_destroy(__FILE__, __LINE__, T)
void _zm_tensor_destroy(const char *file, u32 line, zm_tensor t);

#define zm_tensor_fill(V, ...) _zm_tensor_fill(V, __FILE__, __LINE__, __VA_ARGS__, -1)
zm_tensor _zm_tensor_fill(f32 val, const char *file, u32 line, ...);
#define zm_tensor_zeros(...) _zm_tensor_fill(0, __FILE__, __LINE__, __VA_ARGS__, -1)
#define zm_tensor_ones(...) _zm_tensor_fill(1, __FILE__, __LINE__, __VA_ARGS__, -1)

#define zm_tensor_rand(...) _zm_tensor_rand(__FILE__, __LINE__, __VA_ARGS__, -1)
zm_tensor _zm_tensor_rand(const char *file, u32 line, ...);
#define zm_tensor_randn(...) _zm_tensor_randn(__FILE__, __LINE__, __VA_ARGS__, -1)
zm_tensor _zm_tensor_randn(const char *file, u32 line, ...);
#define zm_tensor_randr(A, B, ...) _zm_tensor_randr(__FILE__, __LINE__, A, B, __VA_ARGS__, -1)
zm_tensor _zm_tensor_randr(const char *file, u32 line, f32 a, f32 b,...);

#define zm_tensor_require_grad(T) _zm_tensor_require_grad(__FILE__, __LINE__, T)
void _zm_tensor_require_grad(char* file, u32 line, zm_tensor *t);

void zm_tensor_backward(zm_tensor *this);

#define zm_tensor_print_data(T) _zm_tensor_print_data((const zm_tensor_view *)T)
void _zm_tensor_print_data(const zm_tensor_view *t);
#define zm_tensor_print_grad(T) _zm_tensor_print_grad((const zm_tensor_view *)T)
void _zm_tensor_print_grad(const zm_tensor_view *t);
#define zm_tensor_print_shape(T) _zm_tensor_print_grad((const zm_tensor_view *)T)
void _zm_tensor_print_shape(const zm_tensor_view *t);

#define zm_tensor_set_prev(T, ...) _zm_tensor_set_prev(__FILE__, __LINE__, T, __VA_ARGS__, NULL)
void _zm_tensor_set_prev(char *file, u32 line, zm_tensor *this, ...);
