#include "znn_util.h"
#include "znn_tensor.h"
#include "znn_random.h"

#include <stdarg.h>

znn_tensor_view znn_tensor_as_tensor_view(znn_tensor *t) {
    znn_tensor_view v = {0};
    memcpy(&v, t, sizeof(v));
    v.is_view = true;
    return v;
}

znn_tensor znn_tensor_view_as_tensor(znn_tensor_view *v) {
    znn_tensor t = {0};
    memcpy(&t, v, sizeof(*v));
    t.is_view = true;
    return t;
}

znn_tensor znn_tensor_from_view(znn_tensor_view *v) {
    znn_tensor t = {0};

    t.is_view = false;
    t.dim = v->dim;
    t.size = v->size;
    t.shape = znn_copy(v->shape, t.dim * 4);
    t.step = znn_copy(v->step, t.dim * 4);
    t.data = znn_copy(v->data, t.size * 4);

    return t;
}

znn_tensor_view _znn_tensor_get_view(znn_tensor_view *t, ...) {
    znn_tensor_view v = *t;
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

znn_tensor_view _znn_tensor_get_range(znn_tensor_view *t, u32 a, u32 b) {
    znn_tensor_view v = *t; // TODO: v.shape
    v.is_view = true;
    v.size = v.size / v.shape[0] * (b - a);
    v.data += a * v.step[0];
    v.grad += v.grad ? a * v.step[0] : 0;
    return v;
}

znn_tensor _znn_tensor_init_v(va_list va1, bool alloc) {
    va_list va2;
    va_copy(va2, va1);
    assert(va_arg(va1, u32) != -1);

    znn_tensor t = {0};
    t.is_view = false;
    for (t.dim = 1; va_arg(va1, u32) != -1; t.dim ++);
    t.shape = znn_malloc(t.dim * 4);
    t.step  = znn_malloc(t.dim * 4);
    t.size = 1;

    for (u32 i = 0; i < t.dim; i ++)
        t.shape[i] = va_arg(va2, u32);

    for (u32 i = 0; i < t.dim; i ++)
        t.step[t.dim - i - 1] = t.size,
        t.size *= t.shape[t.dim - i - 1];

    t.data = alloc ? znn_malloc(t.size * 4) : NULL;

    va_end(va1);
    va_end(va2);
    return t;
}

znn_tensor _znn_tensor_create(const char *file, u32 line, ...) {
    znn_trace(file, line);
    va_list va; va_start(va, line);
    return _znn_tensor_init_v(va, true);
}

znn_tensor _znn_tensor_create_from_data(const char *file, u32 line, void *data, ...) {
    znn_trace(file, line); // TODO: errors
    assert(data);

    va_list va; va_start(va, data);
    znn_tensor t = _znn_tensor_init_v(va, false);
    t.data = data;
    return t;
}

znn_tensor _znn_tensor_create_from_shape(const char *file, u32 line, u32 dim, u32 *shape) {
    znn_trace(file, line); // TODO: errors
    assert(shape);

    znn_tensor t = {0};
    t.is_view = false;
    t.dim = dim;
    t.shape = shape;
    t.step = znn_malloc(t.dim * 4);
    t.size = 1;

    for (u32 i = 0; i < t.dim; i ++)
        t.step[t.dim - i - 1] = t.size,
        t.size *= shape[t.dim - i - 1];

    t.data = znn_malloc(t.size * 4);
    return t;
}

void _znn_tensor_destroy(const char *file, u32 line, znn_tensor t) {
    znn_trace(file, line);
    znn_free(t.grad);
    znn_free(t.data);
    znn_free(t.step);
    znn_free(t.shape);
    if (t.n_prev > 1) znn_free(t.prev);
}

znn_tensor _znn_tensor_fill(f32 val, const char *file, u32 line, ...) {
    znn_trace(file, line);
    va_list va; va_start(va, line);
    znn_tensor t = _znn_tensor_init_v(va, true);
    for (u32 i = 0; i < t.size; i ++) t.data[i] = val;
    return t;
}

znn_tensor _znn_tensor_rand(const char *file, u32 line, ...) {
    znn_trace(file, line);
    va_list va; va_start(va, line);
    znn_tensor t = _znn_tensor_init_v(va, true);
    for (u32 i = 0; i < t.size; i ++) t.data[i] = znn_rand();
    return t;
}

znn_tensor _znn_tensor_randn(const char *file, u32 line, ...) {
    znn_trace(file, line);
    va_list va; va_start(va, line);
    znn_tensor t = _znn_tensor_init_v(va, true);
    for (u32 i = 0; i < t.size; i ++) t.data[i] = znn_randn();
    return t;
}

znn_tensor _znn_tensor_randr(const char *file, u32 line, f32 a, f32 b,...) {
    znn_trace(file, line);
    va_list va; va_start(va, b);
    znn_tensor t = _znn_tensor_init_v(va, true);
    for (u32 i = 0; i < t.size; i ++) t.data[i] = znn_randr(a, b);
    return t;
}

void _znn_tensor_require_grad(char* file, u32 line, znn_tensor *t) {
    znn_trace(file, line);
    if (!t->grad) t->grad = znn_malloc(t->size * 4);
}

void znn_tensor_backward(znn_tensor *this) { // TODO: topological sort
    if (!this->backward) return;
    this->backward(this);

    if (this->n_prev == 1) {
        znn_tensor_backward(this->prev);
    } else if (this->n_prev > 1) {
        znn_tensor **prev = this->prev;
        for (u32 i = 0; i < this->n_prev; i ++)
            znn_tensor_backward(prev[i]);
    }

    if (this->grad)
        memset(this->grad, 0, this->size * 4);
}

static void _znn_tensor_print(const f32 *d, const znn_tensor_view *t, u32 ind, u32 off) {
    if (ind == t->dim) {
        printf("%+10.4f", d[off]);
        return;
    }

    printf("[");
    _znn_tensor_print(d, t, ind + 1, off);
    const u32 S = t->step[ind];
    const u32 T = t->shape[ind] * S;
    if (ind + 1 != t->dim) {
        if (T > S) printf("\n");
        for (int i = S; i < T; i += S) {
            printf("%*s", ind + 1, "");
            _znn_tensor_print(d, t, ind + 1, off + i);
            if (i < T - S) printf("\n");
        }
    } else {
        for (int i = S; i < T; i += S) {
            printf("%*s", ind + 1, "");
            _znn_tensor_print(d, t, ind + 1, off + i);
        }
    }
    printf("]");
}

void _znn_tensor_print_data(const znn_tensor_view *t) {
    if (t->data) {
        _znn_tensor_print(t->data, t, 0, 0);
        printf("\n");
    } else {
        printf("(nil)\n");
    }
}

void _znn_tensor_print_grad(const znn_tensor_view *t) {
    if (t->grad) {
        _znn_tensor_print(t->grad, t, 0, 0);
        printf("\n");
    } else {
        printf("(nil)\n");
    }
}

void _znn_tensor_set_prev(char *file, u32 line, znn_tensor *this, ...) {
    znn_trace(file, line);

    va_list va1, va2;
    va_start(va1, this);
    va_copy(va2, va1);
    assert(va_arg(va1, void*));
    for (this->n_prev = 1; va_arg(va1, void*); this->n_prev ++);

    if (this->n_prev == 1) {
        this->prev = va_arg(va2, void*);
    } else {
        znn_tensor **prev = znn_malloc(this->n_prev * sizeof(void*));
        for (u32 i = 0; i < this->n_prev; i ++)
            prev[i] = va_arg(va2, void*);
        this->prev = prev;
    }

    va_end(va1);
    va_end(va2);
}
