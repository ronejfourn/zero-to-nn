#include "zm_util.h"
#include "zm_layers.h"

#include <assert.h>
#include <string.h>
#include <math.h>

#define L(_L) \
    zm_layer l = {0}; \
    l.forward = zm_layer_forward_ ## _L; \
    l.backward = zm_layer_backward_ ## _L; \
    l.destroy = zm_layer_destroy_ ## _L

ZM_LAYER_FORWARD_FXN(zm_layer_forward_flatten) {
    if (input != this->input) {
        this->input = input;
        u32 shape[] = {input->shape[0], input->_offs[0]};
        zm_tensor_destroy(this->output);
        this->output = zm_tensor_create(2, shape, NULL, true);
    }
    memcpy(this->output.data, input->data, input->_size_flat * sizeof(f32));
}

ZM_LAYER_BACKWARD_FXN(zm_layer_backward_flatten) {
    assert(this->input);
    if (!this->input->grad) return;
    memcpy(this->input->grad, this->output.grad, this->output._size_flat);
}

ZM_LAYER_DESTROY_FXN(zm_layer_destroy_flatten) {
    this.input = NULL;
    zm_tensor_destroy(this.output);
}

zm_layer _zm_layer_flatten(char *file, u32 line) {
    zm_trace(file, line);
    L(flatten);
    return l;
}

typedef struct {
    u32 iter;
    zm_tensor weights;
    zm_tensor biases;
} zm_layer_data_linear;

ZM_LAYER_FORWARD_FXN(zm_layer_forward_linear) {
    zm_layer_data_linear *ld = this->layer_data;
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
        ld->iter = this->output._size_flat / o_f;
    }

    for (u32 k = 0; k < ld->iter; k ++) {
        f32 *ovec = this->output.data + k * o_f;
        f32 *ivec = input->data + k * i_f;
        memcpy(ovec, ld->biases.data, sizeof(f32) * o_f);
        for (int i = 0; i < o_f; i ++) {
            f32 *wvec = ld->weights.data + i * i_f;
            for (int j = 0; j < i_f; j ++) 
                ovec[i] += wvec[j] * ivec[j];
        }
    }
}

ZM_LAYER_BACKWARD_FXN(zm_layer_backward_linear) {
    assert(this->input);
    zm_layer_data_linear *ld = this->layer_data;
    u32 i_f = ld->weights.shape[1];
    u32 o_f = ld->weights.shape[0];
    memset(ld->biases.grad, 0, ld->biases._size_flat);
    memset(ld->weights.grad, 0,ld->weights._size_flat);

    for (u32 k = 0; k < ld->iter; k ++)
        for (u32 i = 0; i < o_f; i ++)
            ld->biases.grad[i] += this->output.grad[k * o_f + i];

    if (this->input->grad) {
        memset(this->input->grad, 0 ,this->input->_size_flat);
        for (u32 k = 0; k < ld->iter; k ++) {
            f32 *ivec = this->input->data + k * i_f;
            f32 *igrad = this->input->grad + k * i_f;
            f32 *ograd = this->output.grad + k * o_f;
            for (u32 i = 0; i < o_f; i ++) {
                f32 *wvec = ld->weights.data + i * i_f;
                f32 *wgrad = ld->weights.grad + i * i_f;
                for (u32 j = 0; j < i_f; j ++) {
                    wgrad[j] += ivec[j] * ograd[i];
                    igrad[j] += wvec[j] * ograd[i];
                }
            }
        }
    } else {
        for (u32 k = 0; k < ld->iter; k ++) {
            f32 *ivec = this->input->data + k * i_f;
            f32 *ograd = this->output.grad + k * o_f;
            for (u32 i = 0; i < o_f; i ++) {
                f32 *wgrad = ld->weights.grad + i * i_f;
                for (u32 j = 0; j < i_f; j ++)
                    wgrad[j] += ivec[j] * ograd[i];
            }
        }
    }
}

ZM_LAYER_DESTROY_FXN(zm_layer_destroy_linear) {
    this.input = NULL;
    zm_tensor_destroy(this.output);
    zm_layer_data_linear *ld = this.layer_data;
    zm_tensor_destroy(ld->weights);
    zm_tensor_destroy(ld->biases);
    zm_free(ld);
}

zm_layer _zm_layer_linear(u32 in_features, u32 out_features, char *file, u32 line) {
    zm_trace(file, line);
    u32 w_shape[] = {out_features, in_features};
    u32 b_shape[] = {out_features};

    zm_layer_data_linear *data = zm_malloc(sizeof(*data));
    data->weights = zm_tensor_randn(2, w_shape);
    zm_tensor_require_grad(&data->weights);
    data->biases  = zm_tensor_randn(1, b_shape);
    zm_tensor_require_grad(&data->biases);

    L(linear);
    l.layer_data = data;

    return l;
}

ZM_LAYER_FORWARD_FXN(zm_layer_forward_ReLU) {
    if (input != this->input) {
        this->input = input;
        zm_tensor_destroy(this->output);
        this->output = zm_tensor_create(input->dim, input->shape, NULL, true);
    }

    for (u32 i = 0; i < input->_size_flat; i ++)
        this->output.data[i] = input->data[i] * (input->data[i] > 0);
}

ZM_LAYER_BACKWARD_FXN(zm_layer_backward_ReLU) {
    assert(this->input);
    if (!this->input->grad) return;
    for (u32 i = 0; i < this->output._size_flat; i ++)
        this->input->grad[i] = 
            (this->input->data[i] > 0) * this->output.grad[i];
}

ZM_LAYER_DESTROY_FXN(zm_layer_destroy_ReLU) {
    this.input = NULL;
    zm_tensor_destroy(this.output);
}

zm_layer _zm_layer_ReLU(char *file, u32 line) {
    zm_trace(file, line);
    L(ReLU);
    return l;
}

typedef struct {
    u32 dim;
} zm_layer_data_softmax;

ZM_LAYER_FORWARD_FXN(zm_layer_forward_softmax) {
    zm_layer_data_softmax *ld = this->layer_data;
    if (input != this->input) {
        this->input = input;
        assert(ld->dim < input->dim);
        zm_tensor_destroy(this->output);
        this->output = zm_tensor_create(input->dim, input->shape, NULL, true);
    }

    zm_tensor *output = &this->output;
    u32 I = ld->dim ? input->_offs[ld->dim - 1] : input->_size_flat;
    u32 J = input->_offs[ld->dim];
    u32 K = input->shape[ld->dim] * J;

    for (int i = 0; i < input->_size_flat; i += I) {
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

ZM_LAYER_BACKWARD_FXN(zm_layer_backward_softmax) {
    assert(this->input);
    if (!this->input->grad) return;

    zm_layer_data_softmax *ld = this->layer_data;
    zm_tensor *input = this->input;
    zm_tensor *output = &this->output;
    memset(input->grad, 0, input->_size_flat);

    u32 I = ld->dim ? input->_offs[ld->dim - 1] : input->_size_flat;
    u32 J = input->_offs[ld->dim];
    u32 K = input->shape[ld->dim] * J;

    for (int i = 0; i < input->_size_flat; i += I) {
        f32 *oi = output->data + i;
        f32 *gi = input->grad + i;
        for (u32 j = 0; j < J; j ++) {
            f32 *oj = oi + j;
            f32 *gj = gi + j;
            for (u32 k = 0; k < K; k += J)
                for (u32 l = 0; l < K; l += J)
                    gj[k] += oj[l] * ((l == k) - oj[k]);
        }
    }
}

ZM_LAYER_DESTROY_FXN(zm_layer_destroy_softmax) {
    this.input = NULL;
    zm_tensor_destroy(this.output);
    zm_free(this.layer_data);
}

zm_layer _zm_layer_softmax(u32 dim, char *file, u32 line) {
    zm_trace(file, line);
    zm_layer_data_softmax *data = zm_malloc(sizeof(*data));
    data->dim = dim;

    L(softmax);
    l.layer_data = data;

    return l;
}

const zm_tensor *zm_sequential_forward(zm_sequential *s, zm_tensor *input) {
    s->layers[0].forward(&s->layers[0], input);
    for (u32 i = 1; i < s->n_layers; i ++)
        s->layers[i].forward(&s->layers[i], &s->layers[i-1].output);
    return &s->layers[s->n_layers - 1].output;
}

void zm_sequential_backward(zm_sequential *s) {
    for (u32 i = 0; i < s->n_layers; i ++)
        s->layers[s->n_layers - i - 1].backward(s->layers + s->n_layers - i - 1);
}

void zm_sequential_destroy(zm_sequential s) {
    for (u32 i = 0; i < s.n_layers; i ++)
        s.layers[i].destroy(s.layers[i]);
    s.n_layers = 0;
    s.layers = NULL;
}
