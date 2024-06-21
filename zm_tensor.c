#include "zm_tensor.h"
#include "zm_random.h"

#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>

zm_tensor zm_tensor_create(u32 _dim, u32 *_shape, void *_data) {
    assert(_dim && _shape);

    zm_tensor t = {0};
    t.dim = _dim;
    t.shape = malloc(_dim);
    t._offs = malloc(_dim);

    t._size_flat = 1;
    for (u32 i = 0; i < _dim; i ++) {
        t._offs[_dim - i - 1] = t._size_flat;
        t._size_flat *= _shape[_dim - i - 1];
        t.shape[i] = _shape[i];
    }

    t.data = malloc(t._size_flat * sizeof(f32));
    if (!_data) return t;

    memcpy(t.data, _data, t._size_flat * sizeof(f32));
    return t;
}

void zm_tensor_destroy(zm_tensor t) {
    free(t.data);
    free(t.shape);
    free(t._offs);
}

zm_tensor zm_tensor_zeros(u32 _dim, u32 *_shape) {
    zm_tensor t = zm_tensor_create(_dim, _shape, NULL);
    for (u32 i = 0; i < t._size_flat; i ++)
        t.data[i] = 0;
    return t;
}

zm_tensor zm_tensor_ones(u32 _dim, u32 *_shape) {
    zm_tensor t = zm_tensor_create(_dim, _shape, NULL);
    for (u32 i = 0; i < t._size_flat; i ++)
        t.data[i] = 1;
    return t;
}

zm_tensor zm_tensor_random(u32 _dim, u32 *_shape) {
    zm_tensor t = zm_tensor_create(_dim, _shape, NULL);
    for (u32 i = 0; i < t._size_flat; i ++)
        t.data[i] = zm_random();
    return t;
}

zm_tensor zm_tensor_random_n(u32 _dim, u32 *_shape) {
    zm_tensor t = zm_tensor_create(_dim, _shape, NULL);
    for (u32 i = 0; i < t._size_flat; i ++)
        t.data[i] = zm_random_n();
    return t;
}

void _zm_tensor_print(const zm_tensor *t, u32 ind, u32 off) {
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
