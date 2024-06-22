#include "common.h"

#include <math.h>

typedef struct {
    u32 dim;
} zm_layer_data_softmax;

ZM_TENSOR_BACKWARD_FXN(zm_layer_backward_softmax) {
    zm_tensor *prev = this->prev;
    if (!prev->grad) return;

    uintptr_t dim = (uintptr_t)this->backward_data;
    u32 I = dim ? prev->step[dim - 1] : prev->size;
    u32 J = prev->step[dim];
    u32 K = prev->shape[dim] * J;

    for (int i = 0; i < prev->size; i += I) {
        f32 *oi = this->data + i;
        f32 *gi = prev->grad + i;
        for (u32 j = 0; j < J; j ++) {
            f32 *oj = oi + j;
            f32 *gj = gi + j;
            for (u32 k = 0; k < K; k += J)
                for (u32 l = 0; l < K; l += J)
                    gj[k] += oj[l] * ((l == k) - oj[k]);
        }
    }
}

ZM_LAYER_FORWARD_FXN(zm_layer_forward_softmax) {
    zm_layer_data_softmax *ld = this->data;
    if (input != this->input) {
        assert(ld->dim < input->dim);
        _update(this, input, zm_layer_backward_softmax,
                zm_tensor_create_from_shape(input->dim, zm_copy(input->shape, input->dim * 4)));
        zm_tensor_set_prev(&this->output, input, 1);
        this->output.backward_data = (void *)(uintptr_t)ld->dim;
    }

    zm_tensor *output = &this->output;
    u32 I = ld->dim ? input->step[ld->dim - 1] : input->size;
    u32 J = input->step[ld->dim];
    u32 K = input->shape[ld->dim] * J;

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

ZM_LAYER_DESTROY_FXN(zm_layer_destroy_softmax) {
    zm_tensor_destroy(this.output);
    zm_free(this.data);
}

zm_layer _zm_layer_softmax(u32 dim, char *file, u32 line) {
    zm_trace(file, line);
    zm_layer_data_softmax *data = zm_malloc(sizeof(*data));
    data->dim = dim;

    L(softmax);
    l.data = data;

    return l;
}
