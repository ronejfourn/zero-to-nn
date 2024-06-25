#include "znn_util.h"
#include "znn_tensor.h"
#include "znn_random.h"

#include <stdarg.h>

static znn_tensor _znn__tensor_init(va_list va1, bool alloc) {
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

static void _znn__tensor_print(const znn_tensor_view *t, u32 ind, u32 off) {
    if (ind == t->dim) {
        printf("%+10.4f", t->data[off]);
        return;
    }

    printf("[");
    _znn__tensor_print(t, ind + 1, off);
    const u32 S = t->step[ind];
    const u32 T = t->shape[ind] * S;
    if (ind + 1 != t->dim) {
        if (T > S) printf("\n");
        for (int i = S; i < T; i += S) {
            printf("%*s", ind + 1, "");
            _znn__tensor_print(t, ind + 1, off + i);
            if (i < T - S) printf("\n");
        }
    } else {
        for (int i = S; i < T; i += S) {
            printf("%*s", ind + 1, "");
            _znn__tensor_print(t, ind + 1, off + i);
        }
    }
    printf("]");
}

void _znn_tensor_divide(znn_tensor_view *t, f32 s) {
    for (u32 i = 0; i < t->size; i ++)
        t->data[i] /= s;
}

znn_tensor _znn_tensor_one_hot(const char *file, u32 line, znn_tensor_view *input, u32 num_classes) {
    znn_trace(file, line);

    u32 *shape = znn_malloc((input->dim + 1) * 4);
    memcpy(shape, input->shape, input->dim * 4);
    shape[input->dim] = num_classes;

    znn_tensor output = znn_tensor_from_shape(input->dim + 1, shape);

    for (u32 i = 0; i < input->size; i ++) {
        i64 o = (i64)input->data[i];
        assert(input->data[i] == o && (o >= 0 && o < num_classes));
        output.data[i * num_classes + o] = 1;
    }

    return output;
}

znn_tensor_view _znn_tensor_get(znn_tensor_view *t, ...) {
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

znn_tensor _znn_tensor_create(const char *file, u32 line, ...) {
    znn_trace(file, line);
    va_list va; va_start(va, line);
    return _znn__tensor_init(va, true);
}

znn_tensor _znn_tensor_from_data(const char *file, u32 line, void *data, ...) {
    znn_trace(file, line);

    va_list va; va_start(va, data);
    znn_tensor t = _znn__tensor_init(va, false);
    t.data = data;
    return t;
}

znn_tensor _znn_tensor_from_shape(const char *file, u32 line, u32 dim, u32 *shape) {
    znn_trace(file, line);
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

znn_tensor _znn_tensor_from_view(const char *file, u32 line, znn_tensor_view *v) {
    znn_trace(file, line);
    znn_tensor t = {0};

    t.is_view = false;
    t.dim = v->dim;
    t.size = v->size;
    t.shape = znn_copy(v->shape, t.dim * 4);
    t.step = znn_copy(v->step, t.dim * 4);
    t.data = znn_copy(v->data, t.size * 4);

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
    znn_tensor t = _znn__tensor_init(va, true);
    for (u32 i = 0; i < t.size; i ++) t.data[i] = val;
    return t;
}

znn_tensor _znn_tensor_rand(const char *file, u32 line, ...) {
    znn_trace(file, line);
    va_list va; va_start(va, line);
    znn_tensor t = _znn__tensor_init(va, true);
    for (u32 i = 0; i < t.size; i ++) t.data[i] = znn_rand();
    return t;
}

znn_tensor _znn_tensor_randn(const char *file, u32 line, ...) {
    znn_trace(file, line);
    va_list va; va_start(va, line);
    znn_tensor t = _znn__tensor_init(va, true);
    for (u32 i = 0; i < t.size; i ++) t.data[i] = znn_randn();
    return t;
}

znn_tensor _znn_tensor_randr(const char *file, u32 line, f32 a, f32 b,...) {
    znn_trace(file, line);
    va_list va; va_start(va, b);
    znn_tensor t = _znn__tensor_init(va, true);
    for (u32 i = 0; i < t.size; i ++) t.data[i] = znn_randr(a, b);
    return t;
}

void _znn_tensor_require_grad(char* file, u32 line, znn_tensor *t) {
    znn_trace(file, line);
    if (!t->grad) t->grad = znn_malloc(t->size * 4);
}

void znn_tensor_backward(znn_tensor *this) {
    if (!this->backward.fn) return;
    this->backward.fn(this);
    znn_tensor_backward_fxn p = this->backward.fn;
    this->backward.fn = NULL;

    if (this->n_prev == 1) {
        znn_tensor_backward(this->prev);
    } else if (this->n_prev > 1) {
        znn_tensor **prev = this->prev;
        for (u32 i = 0; i < this->n_prev; i ++)
            znn_tensor_backward(prev[i]);
    }

    if (this->grad)
        memset(this->grad, 0, this->size * 4);
    this->backward.fn = p;
}

void _znn_tensor_print(const znn_tensor_view *v, ...) {
    bool print_data = true, print_grad = false, print_shape = false;

    va_list va;
    va_start(va, v);
    for (u32 opt = va_arg(va, u32); opt != _ZNN_TENSOR_PRINT_END; opt = va_arg(va, u32)) {
        switch (opt) {
        case ZNN_TENSOR_PRINT_DATA    : print_data  = true;  break; 
        case ZNN_TENSOR_PRINT_GRAD    : print_grad  = true;  break;
        case ZNN_TENSOR_PRINT_SHAPE   : print_shape = true;  break;
        case ZNN_TENSOR_PRINT_NO_DATA : print_data  = false; break;
        case ZNN_TENSOR_PRINT_NO_GRAD : print_grad  = false; break;
        case ZNN_TENSOR_PRINT_NO_SHAPE: print_shape = false; break;
        default: znn_unreachable();
        }
    }
    va_end(va);

    if (print_data) {
        _znn__tensor_print(v, 0, 0);
        printf("\n");
    }

    if (print_shape) {
        printf("[ ");
        for (u32 i = 0; i < v->dim; i ++)
            printf("%5d", v->shape[i]);
        printf(" ]\n");
    }

    if (print_grad) {
        znn_tensor_view a = znn_tensor_get(v);
        a.data = a.grad;
        _znn__tensor_print(&a, 0, 0);
        printf("\n");
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
