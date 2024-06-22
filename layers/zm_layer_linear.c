#include "common.h"

typedef struct {
    zm_tensor weights;
    zm_tensor biases;
} zm_layer_data_linear;

ZM_LAYER_ZERO_GRAD_FXN(zm_layer_zero_grad_linear) {
    zm_layer_data_linear *ld = this->data;
    memset(ld->biases.grad, 0, ld->biases._size_flat * sizeof(f32));
    memset(ld->weights.grad, 0, ld->weights._size_flat * sizeof(f32));
}

ZM_TENSOR_BACKWARD_FXN(zm_layer_backward_linear) {
    zm_tensor **prev = this->prev;
    zm_tensor *input = prev[0];
    zm_tensor *weights = prev[1];
    zm_tensor *biases = prev[2];

    u32 i_f = weights->shape[1];
    u32 o_f = weights->shape[0];

    u32 iter = this->_size_flat / o_f;
    for (u32 k = 0; k < iter; k ++)
        for (u32 i = 0; i < o_f; i ++)
            biases->grad[i] += this->grad[k * o_f + i];

    if (input->grad) {
        for (u32 k = 0; k < iter; k ++) {
            f32 *ivec = input->data + k * i_f;
            f32 *igrad = input->grad + k * i_f;
            f32 *ograd = this->grad + k * o_f;
            for (u32 i = 0; i < o_f; i ++) {
                f32 *wvec = weights->data + i * i_f;
                f32 *wgrad = weights->grad + i * i_f;
                for (u32 j = 0; j < i_f; j ++) {
                    wgrad[j] += ivec[j] * ograd[i];
                    igrad[j] += wvec[j] * ograd[i];
                }
            }
        }
    } else {
        for (u32 k = 0; k < iter; k ++) {
            f32 *ivec = input->data + k * i_f;
            f32 *ograd = this->grad + k * o_f;
            for (u32 i = 0; i < o_f; i ++) {
                f32 *wgrad = weights->grad + i * i_f;
                for (u32 j = 0; j < i_f; j ++)
                    wgrad[j] += ivec[j] * ograd[i];
            }
        }
    }
}

ZM_LAYER_FORWARD_FXN(zm_layer_forward_linear) {
    zm_layer_data_linear *ld = this->data;
    u32 i_f = ld->weights.shape[1];
    u32 o_f = ld->weights.shape[0];
    assert(input->shape[input->dim - 1] == i_f);

    if (input != this->input) {
        this->input = input;
        u32 *shape = zm_copy(input->shape, input->dim * sizeof(*input->shape));
        shape[input->dim - 1] = o_f;
        zm_tensor_destroy(this->output);
        this->output = zm_tensor_create(input->dim, shape, NULL, true);
        zm_free(shape);

        zm_tensor *p[] = {input, &ld->weights, &ld->biases};
        zm_tensor_set_prev(&this->output, p, 3);
        this->output.backward = zm_layer_backward_linear;
    }

    u32 iter = this->output._size_flat / o_f;
    for (u32 k = 0; k < iter; k ++) {
        f32 *ovec = this->output.data + k * o_f;
        f32 *ivec = input->data + k * i_f;
        memcpy(ovec, ld->biases.data, sizeof(f32) * o_f);
        for (u32 i = 0; i < o_f; i ++) {
            f32 *wvec = ld->weights.data + i * i_f;
            for (u32 j = 0; j < i_f; j ++)
                ovec[i] += wvec[j] * ivec[j];
        }
    }
}

ZM_LAYER_DESTROY_FXN(zm_layer_destroy_linear) {
    zm_tensor_destroy(this.output);
    zm_layer_data_linear *ld = this.data;
    zm_tensor_destroy(ld->weights);
    zm_tensor_destroy(ld->biases);
    zm_free(ld);
}

ZM_LAYER_SHD_FXN(zm_layer_shd_linear) {
    zm_layer_data_linear *ld = this->data;

    zm_tensor *weights = &ld->weights;
    zm_tensor *biases = &ld->biases;

    for (u32 i = 0; i < weights->_size_flat; i++)
        weights->data[i] -= lr * weights->grad[i];

    for (u32 i = 0; i < biases->_size_flat; i++)
        biases->data[i] -= lr * biases->grad[i];
}

zm_layer _zm_layer_linear(u32 in_features, u32 out_features, char *file, u32 line) {
    zm_trace(file, line);
    u32 w_shape[] = {out_features, in_features};
    u32 b_shape[] = {out_features};

    zm_layer_data_linear *data = zm_malloc(sizeof(*data));
    data->weights = zm_tensor_randn(2, w_shape);
    zm_tensor_require_grad(&data->weights);
    data->biases  = zm_tensor_ones(1, b_shape);
    zm_tensor_require_grad(&data->biases);

    L(linear);
    l.data = data;
    l.shd = zm_layer_shd_linear;
    l.zero_grad = zm_layer_zero_grad_linear;

    return l;
}
