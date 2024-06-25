#include "common.h"

#if ZNN_OPENMP_ENABLE
#include <omp.h>
#endif

static ZNN_TENSOR_BACKWARD_FXN(_znn__backward_softmax) {
    uintptr_t dim = (uintptr_t)this->backward_data;
    znn_tensor *prev = this->prev;
    u32 I = dim ? prev->step[dim - 1] : prev->size;
    u32 J = prev->step[dim];
    u32 K = prev->shape[dim] * J;

#if ZNN_OPENMP_ENABLE
    #pragma omp parallel for
#endif
    for (int i = 0; i < prev->size; i += I) {
        f32 *oi = this->data + i, *gi = prev->grad + i;
        for (u32 j = 0; j < J; j ++) {
            f32 *oj = oi + j, *gj = gi + j;
            for (u32 k = 0; k < K; k += J)
                for (u32 l = 0; l < K; l += J)
                    gj[k] += oj[l] * ((l == k) - oj[k]);
        }
    }
}

static ZNN_LAYER_INIT_FXN(_znn__init_softmax) {
    u32 dim = (u32)(uintptr_t)this->parameters;
    assert(dim < input->dim);
    this->output = znn_tensor_from_shape(
            input->dim, znn_copy(input->shape, input->dim * 4));
    if (input->grad)
        ZNN_FXN_SET(this->output.backward, _znn__backward_softmax);
    this->output.backward_data = (void *)(uintptr_t)dim;
    znn_tensor_require_grad(&this->output);
    znn_tensor_set_prev(&this->output, input);
}

static ZNN_LAYER_FORWARD_FXN(_znn__forward_softmax) {
    u32 dim = (u32)(uintptr_t)this->parameters;
    znn_tensor *input = this->input;
    znn_tensor *output = &this->output;

    u32 I = dim ? input->step[dim - 1] : input->size;
    u32 J = input->step[dim];
    u32 K = input->shape[dim] * J;

#if ZNN_OPENMP_ENABLE
    #pragma omp parallel for
#endif
    for (int i = 0; i < input->size; i += I) {
        f32 *oi = output->data + i;
        for (u32 j = 0; j < J; j ++) {
            f32 sum = 0;
            f32 *oj = oi + j;
            for (u32 k = 0; k < K; k += J)
                sum += (oj[k] = exp(input->data[i + j + k]));

            for (u32 k = 0; k < K; k += J)
                oj[k] /= sum;
        }
    }
}

static ZNN_LAYER_DESTROY_FXN(_znn__destroy_softmax) {
    this->output.backward_data = NULL;
}

znn_layer _znn_layer_softmax(u32 dim, char *file, u32 line) {
    znn_trace(file, line);

    znn_layer l = {0};
    ZNN_FXN_SET(l.init, _znn__init_softmax);
    ZNN_FXN_SET(l.forward, _znn__forward_softmax);
    ZNN_FXN_SET(l.destroy, _znn__destroy_softmax);
    l.parameters = (void*)(uintptr_t)dim;

    return l;
}
