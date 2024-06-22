#include "zm_util.h"
#include "zm_tensor.h"
#include "zm_random.h"

#include <stdio.h>
#include <string.h>
#include <assert.h>

zm_tensor _zm_tensor_create(u32 _dim, u32 *_shape, void *_data, bool require_grad, const char *file, u32 line) {
    zm_trace(file, line);
    assert(_dim && _shape);

    zm_tensor t = {0};
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
    if (require_grad)
        t.grad = zm_malloc(t._size_flat * sizeof(f32));

    return t;
}

void _zm_tensor_destroy(zm_tensor t, const char *file, u32 line) {
    zm_trace(file, line);
    zm_free(t.grad);
    zm_free(t.data);
    zm_free(t._offs);
    zm_free(t.shape);
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

static void _zm_tensor_print(const zm_tensor *t, u32 ind, u32 off) {
    if (ind == t->dim) {
        printf("%+9.6f  ", t->data[off]);
        return;
    }

    if (ind + 1 == t->dim)
        printf("%*s[  ", ind * 2, "");
    else
        printf("%*s[\n", ind * 2, "");
    for (int i = 0; i < t->shape[ind]; i ++)
        _zm_tensor_print(t, ind + 1, off + i * t->_offs[ind]);
    if (ind + 1 == t->dim)
        printf("]\n");
    else
        printf("%*s]\n", ind * 2, "");
}

void zm_tensor_print(const zm_tensor *t) {
    _zm_tensor_print(t, 0, 0);
}
