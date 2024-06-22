#include "zm_util.h"
#include "zm_tensor.h"
#include "zm_random.h"

#include <stdio.h>
#include <string.h>
#include <assert.h>

ZM_TENSOR_BACKWARD_FXN(_back_dummy) {}

zm_tensor _zm_tensor_create(u32 _dim, u32 *_shape, void *_data, bool require_grad, const char *file, u32 line) {
    zm_trace(file, line);
    assert(_dim && _shape);

    zm_tensor t;
    t.dim = _dim;
    t.shape = zm_malloc(_dim * sizeof(u32));
    t._offs = zm_malloc(_dim * sizeof(u32));
    t._size_flat = 1;

    for (u32 i = 0; i < _dim; i ++) {
        t._offs[_dim - i - 1] = t._size_flat;
        t._size_flat *= _shape[_dim - i - 1];
        t.shape[i] = _shape[i];
    }

    t.data = zm_copy(_data, t._size_flat * sizeof(f32));
    t.grad = require_grad ? zm_malloc(t._size_flat * sizeof(f32)) : NULL;
    t.backward = _back_dummy;
    t.n_prev = 0;

    return t;
}

void _zm_tensor_destroy(zm_tensor t, const char *file, u32 line) {
    zm_trace(file, line);
    zm_free(t.grad);
    zm_free(t.data);
    zm_free(t._offs);
    zm_free(t.shape);
    if (t.n_prev > 1) zm_free(t.prev);
}

zm_tensor _zm_tensor_fill(u32 _dim, u32 *_shape, f32 v, char *file, u32 line) {
    zm_tensor t = _zm_tensor_create(_dim, _shape, NULL, false, file, line);
    for (u32 i = 0; i < t._size_flat; i ++)
        t.data[i] = v;
    return t;
}

zm_tensor _zm_tensor_rand(u32 _dim, u32 *_shape, char *file, u32 line) {
    zm_tensor t = _zm_tensor_create(_dim, _shape, NULL, false, file, line);
    for (u32 i = 0; i < t._size_flat; i ++)
        t.data[i] = zm_rand();
    return t;
}

zm_tensor _zm_tensor_randn(u32 _dim, u32 *_shape, char *file, u32 line) {
    zm_tensor t = _zm_tensor_create(_dim, _shape, NULL, false, file, line);
    for (u32 i = 0; i < t._size_flat; i ++)
        t.data[i] = zm_randn();
    return t;
}

void _zm_tensor_require_grad(zm_tensor *t, char* file, u32 line) {
    if (!t->grad) {
        zm_trace(file, line);
        t->grad = zm_malloc(t->_size_flat * sizeof(f32));
    }
}

void zm_tensor_backward(zm_tensor *this) { // TODO: topological sort
    if (!this) return;
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
    const u32 S = t->_offs[ind];
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

void _zm_tensor_set_prev(zm_tensor *t, void *p, u32 n, char *file, u32 line) {
    zm_trace(file, line);
    t->n_prev = n;
    t->prev = (n == 1) ? p : zm_copy(p, sizeof(void*) * n);
}
