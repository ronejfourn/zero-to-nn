#include "zm_util.h"
#include "zm_tensor.h"
#include "zm_random.h"

#include <stdarg.h>

zm_tensor_view zm_tensor_as_tensor_view(zm_tensor *t) {
    zm_tensor_view v = {0};
    memcpy(&v, t, sizeof(v));
    v.is_view = true;
    return v;
}

zm_tensor zm_tensor_view_as_tensor(zm_tensor_view *v) {
    zm_tensor t = {0};
    memcpy(&t, v, sizeof(*v));
    t.is_view = true;
    return t;
}

zm_tensor zm_tensor_from_view(zm_tensor_view *v) {
    zm_tensor t = {0};

    t.is_view = false;
    t.dim = v->dim;
    t.shape = zm_copy(v->shape, t.dim * 4);
    t.step = zm_copy(v->step, t.dim * 4);
    t.size = v->size;

    t.data_base = zm_malloc(t.size * 4 + 16);
    t.data = zm_alignptr16(t.data_base);
    memcpy(t.data, v->data, t.size * 4);

    return t;
}

zm_tensor_view _zm_tensor_get_view(zm_tensor_view *t, ...) {
    zm_tensor_view v = *t;
    v.is_view = true;

    va_list v1;
    va_start(v1, t);

    for (u32 i = 0, s = va_arg(v1, u32); s != -1; i++, s = va_arg(v1, u32)) {
        assert(i < t->dim && s < t->shape[i]);
        v.dim   -= 1;
        v.shape = v.dim ? v.shape + 1 : NULL;
        v.step  = v.dim ? v.shape + 1 : NULL;
        v.size  = v.dim ? v.shape[0] : 0;
        v.data  += s * t->step[i];
        v.grad  += v.grad ? s * t->step[i] : 0;
    }

    va_end(v1);
    return v;
}

zm_tensor_view _zm_tensor_get_range(zm_tensor_view *t, u32 a, u32 b) {
    zm_tensor_view v = *t; // TODO: v.shape
    v.is_view = true;
    v.size = v.size / v.shape[0] * (b - a);
    v.data += a * v.step[0];
    v.grad += v.grad ? a * v.step[0] : 0;
    return v;
}

zm_tensor _zm_tensor_init_v(va_list va1) {
    va_list va2;
    va_copy(va2, va1);
    assert(va_arg(va1, u32) != -1);

    zm_tensor t = {0};
    t.is_view = false;
    for (t.dim = 1; va_arg(va1, u32) != -1; t.dim ++);
    t.shape = zm_malloc(t.dim * 4);
    t.step  = zm_malloc(t.dim * 4);
    t.size = 1;

    for (u32 i = 0; i < t.dim; i ++)
        t.shape[i] = va_arg(va2, u32);

    for (u32 i = 0; i < t.dim; i ++)
        t.step[t.dim - i - 1] = t.size,
        t.size *= t.shape[t.dim - i - 1];

    va_end(va1);
    va_end(va2);
    return t;
}

zm_tensor _zm_tensor_create(const char *file, u32 line, ...) {
    zm_trace(file, line);
    va_list va; va_start(va, line);
    zm_tensor t = _zm_tensor_init_v(va);
    t.data_base = zm_malloc(t.size * 4 + 16);
    t.data = zm_alignptr16(t.data_base);
    return t;
}

zm_tensor _zm_tensor_create_from_data(const char *file, u32 line, void *data, ...) {
    zm_trace(file, line); // TODO: errors
    assert(data);

    va_list va; va_start(va, data);
    zm_tensor t = _zm_tensor_init_v(va);
    if (((uintptr_t)data & 15) == 0) {
        t.data_base = data;
        t.data = data;
    } else {
        t.data_base = zm_malloc(t.size * 4 + 16);
        t.data = zm_alignptr16(t.data_base);
        memcpy(t.data, data, t.size * 4);
        zm_free(data);
    }
    return t;
}

zm_tensor _zm_tensor_create_from_shape(const char *file, u32 line, u32 dim, u32 *shape) {
    zm_trace(file, line); // TODO: errors
    assert(shape);

    zm_tensor t = {0};
    t.is_view = false;
    t.dim = dim;
    t.shape = shape;
    t.step = zm_malloc(t.dim * 4);
    t.size = 1;

    for (u32 i = 0; i < t.dim; i ++)
        t.step[t.dim - i - 1] = t.size,
        t.size *= shape[t.dim - i - 1];

    t.data_base = zm_malloc(t.size * 4 + 16);
    t.data = zm_alignptr16(t.data_base);
    return t;
}

void _zm_tensor_destroy(const char *file, u32 line, zm_tensor t) {
    zm_trace(file, line);
    zm_free(t.grad_base);
    zm_free(t.data_base);
    zm_free(t.step);
    zm_free(t.shape);
    if (t.n_prev > 1) zm_free(t.prev);
}

zm_tensor _zm_tensor_fill(f32 val, const char *file, u32 line, ...) {
    zm_trace(file, line);
    va_list va; va_start(va, line);
    zm_tensor t = _zm_tensor_init_v(va);
    t.data_base = zm_malloc(t.size * 4 + 16);
    t.data = zm_alignptr16(t.data_base);
    for (u32 i = 0; i < t.size; i ++) t.data[i] = val;
    return t;
}

zm_tensor _zm_tensor_rand(const char *file, u32 line, ...) {
    zm_trace(file, line);
    va_list va; va_start(va, line);
    zm_tensor t = _zm_tensor_init_v(va);
    t.data_base = zm_malloc(t.size * 4 + 16);
    t.data = zm_alignptr16(t.data_base);
    for (u32 i = 0; i < t.size; i ++) t.data[i] = zm_rand();
    return t;
}

zm_tensor _zm_tensor_randn(const char *file, u32 line, ...) {
    zm_trace(file, line);
    va_list va; va_start(va, line);
    zm_tensor t = _zm_tensor_init_v(va);
    t.data_base = zm_malloc(t.size * 4 + 16);
    t.data = zm_alignptr16(t.data_base);
    for (u32 i = 0; i < t.size; i ++) t.data[i] = zm_randn();
    return t;
}

zm_tensor _zm_tensor_randr(const char *file, u32 line, f32 a, f32 b,...) {
    zm_trace(file, line);
    va_list va; va_start(va, b);
    zm_tensor t = _zm_tensor_init_v(va);
    t.data_base = zm_malloc(t.size * 4 + 16);
    t.data = zm_alignptr16(t.data_base);
    for (u32 i = 0; i < t.size; i ++) t.data[i] = zm_randr(a, b);
    return t;
}

void _zm_tensor_require_grad(char* file, u32 line, zm_tensor *t) {
    if (!t->grad) {
        zm_trace(file, line);
        t->grad_base = zm_malloc(t->size * 4 + 16);
        t->grad = zm_alignptr16(t->grad_base);
    }
}

void zm_tensor_backward(zm_tensor *this) { // TODO: topological sort
    if (!this->backward) return;
    this->backward(this);

    if (this->n_prev == 1) {
        zm_tensor_backward(this->prev);
    } else if (this->n_prev > 1) {
        zm_tensor **prev = this->prev;
        for (u32 i = 0; i < this->n_prev; i ++)
            zm_tensor_backward(prev[i]);
    }

    if (this->grad)
        memset(this->grad, 0, this->size * 4);
}

static void _zm_tensor_print(const f32 *d, const zm_tensor_view *t, u32 ind, u32 off) {
    if (ind == t->dim) {
        printf("%+10.4f", d[off]);
        return;
    }

    printf("[");
    _zm_tensor_print(d, t, ind + 1, off);
    const u32 S = t->step[ind];
    const u32 T = t->shape[ind] * S;
    if (ind + 1 != t->dim) {
        if (T > S) printf("\n");
        for (int i = S; i < T; i += S) {
            printf("%*s", ind + 1, "");
            _zm_tensor_print(d, t, ind + 1, off + i);
            if (i < T - S) printf("\n");
        }
    } else {
        for (int i = S; i < T; i += S) {
            printf("%*s", ind + 1, "");
            _zm_tensor_print(d, t, ind + 1, off + i);
        }
    }
    printf("]");
}

void _zm_tensor_print_data(const zm_tensor_view *t) {
    if (t->data) {
        _zm_tensor_print(t->data, t, 0, 0);
        printf("\n");
    } else {
        printf("(nil)\n");
    }
}

void _zm_tensor_print_grad(const zm_tensor_view *t) {
    if (t->grad) {
        _zm_tensor_print(t->grad, t, 0, 0);
        printf("\n");
    } else {
        printf("(nil)\n");
    }
}

void _zm_tensor_set_prev(char *file, u32 line, zm_tensor *this, ...) {
    zm_trace(file, line);

    va_list va1, va2;
    va_start(va1, this);
    va_copy(va2, va1);
    assert(va_arg(va1, void*));
    for (this->n_prev = 1; va_arg(va1, void*); this->n_prev ++);

    if (this->n_prev == 1) {
        this->prev = va_arg(va2, void*);
    } else {
        zm_tensor **prev = zm_malloc(this->n_prev * sizeof(void*));
        for (u32 i = 0; i < this->n_prev; i ++)
            prev[i] = va_arg(va2, void*);
        this->prev = prev;
    }

    va_end(va1);
    va_end(va2);
}
