#include "zm_util.h"
#include "zm_tensor.h"
#include "zm_random.h"

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdarg.h>

zm_tensor _zm_tensor_create_v(const char *file, u32 line, va_list va1) {
    zm_trace(file, line);

    va_list va2;
    va_copy(va2, va1);
    assert(va_arg(va1, u32) != -1);

    zm_tensor t = {0};
    for (t.dim = 1; va_arg(va1, u32) != -1; t.dim ++);
    t.shape = zm_malloc(t.dim * 4);
    t.step  = zm_malloc(t.dim * 4);
    t.size = 1;

    for (u32 i = 0; i < t.dim; i ++)
        t.shape[i] = va_arg(va2, u32);

    for (u32 i = 0; i < t.dim; i ++)
        t.step[t.dim - i - 1] = t.size,
        t.size *= t.shape[t.dim - i - 1];

    t.data = zm_malloc(t.size * 4);

    va_end(va1);
    va_end(va2);
    return t;
}

zm_tensor _zm_tensor_create(const char *file, u32 line, ...) {
    va_list va;
    va_start(va, line);
    return _zm_tensor_create_v(file, line, va);
}

zm_tensor _zm_tensor_create_from_shape(const char *file, u32 line, u32 dim, u32 *shape) {
    zm_trace(file, line); // TODO: errors

    zm_tensor t = {0};
    t.dim = dim;
    t.shape = shape;
    t.step  = zm_malloc(t.dim * 4);
    t.size = 1;

    for (u32 i = 0; i < t.dim; i ++)
        t.step[t.dim - i - 1] = t.size,
        t.size *= shape[t.dim - i - 1];

    t.data = zm_malloc(t.size * 4);
    t.grad = NULL;
    return t;
}

zm_tensor _zm_tensor_create_from_data(const char *file, u32 line, u32 dim, u32 *shape, void *data) {
    zm_tensor t = _zm_tensor_create_from_shape(file, line, dim, shape);
    memcpy(t.data, data, t.size * 4);
    return t;
}

void _zm_tensor_destroy(const char *file, u32 line, zm_tensor t) {
    zm_trace(file, line);
    zm_free(t.grad);
    zm_free(t.data);
    zm_free(t.step);
    zm_free(t.shape);
    if (t.n_prev > 1) zm_free(t.prev);
}

zm_tensor _zm_tensor_fill(const char *file, u32 line, f32 val, ...) {
    va_list va;
    va_start(va, val);
    zm_tensor t = _zm_tensor_create_v(file, line, va);
    for (u32 i = 0; i < t.size; i ++) t.data[i] = val;
    return t;
}

zm_tensor _zm_tensor_rand(const char *file, u32 line, ...) {
    va_list va;
    va_start(va, line);
    zm_tensor t = _zm_tensor_create_v(file, line, va);
    for (u32 i = 0; i < t.size; i ++) t.data[i] = zm_rand();
    return t;
}

zm_tensor _zm_tensor_randn(const char *file, u32 line, ...) {
    va_list va;
    va_start(va, line);
    zm_tensor t = _zm_tensor_create_v(file, line, va);
    for (u32 i = 0; i < t.size; i ++) t.data[i] = zm_randn();
    return t;
}

void _zm_tensor_require_grad(char* file, u32 line, zm_tensor *t) {
    if (!t->grad) {
        zm_trace(file, line);
        t->grad = zm_malloc(t->size * 4);
    }
}

void zm_tensor_backward(zm_tensor *this) { // TODO: topological sort
    if (!this || !this->backward) return;
    this->backward(this);

    if (this->n_prev == 1) {
        zm_tensor_backward(this->prev);
    } else if (this->n_prev > 1) {
        zm_tensor **prev = this->prev;
        for (u32 i = 0; i < this->n_prev; i ++)
            zm_tensor_backward(prev[i]);
    }
}

static void _zm_tensor_print(const f32 *d, const zm_tensor *t, u32 ind, u32 off) {
    if (ind == t->dim) {
        printf("%+9.6f  ", d[off]);
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

void zm_tensor_print_data(const zm_tensor *t) {
    if (t->data) {
        _zm_tensor_print(t->data, t, 0, 0);
        printf("\n");
    } else {
        printf("(nil)\n");
    }
}

void zm_tensor_print_grad(const zm_tensor *t) {
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
