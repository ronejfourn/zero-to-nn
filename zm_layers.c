#include "zm_util.h"
#include "zm_layers.h"

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>

typedef struct {
    u32 iter;
    zm_tensor weights;
    zm_tensor biases;
} zm_layer_data_linear;

void zm_layer_forward_linear(zm_layer *this, const zm_tensor *input) {
    zm_layer_data_linear *ld = this->layer_data;
    u32 i_f = ld->weights.shape[1];
    u32 o_f = ld->weights.shape[0];
    assert(input->shape[input->dim - 1] == i_f);

    if (input != this->input) {
        zm_tensor_destroy(this->output);
        u32 *shape = zm_copy(input->shape, input->dim * sizeof(*input->shape));
        shape[input->dim - 1] = o_f;
        this->output = zm_tensor_create(input->dim, shape, NULL);
        free(shape);
        ld->iter = this->output._size_flat / o_f;
    }

    for (u32 k = 0; k < ld->iter; k ++) {
        f32 *ovec = this->output.data + k * o_f;
        f32 *ivec = input->data + k * i_f;
        memcpy(ovec, ld->biases.data, sizeof(f32) * o_f);
        for (int i = 0; i < o_f; i ++) {
            u32 is = i * i_f;
            for (int j = 0; j < i_f; j ++)
                ovec[i] += ld->weights.data[is + j] * ivec[j];
        }
    }
}

zm_layer zm_layer_linear(u32 in_features, u32 out_features) {
    u32 w_shape[] = {out_features, in_features};
    u32 b_shape[] = {out_features};

    zm_layer_data_linear *data = malloc(sizeof(*data));
    data->weights = zm_tensor_random_n(2, w_shape);
    data->biases  = zm_tensor_random_n(1, b_shape);

    zm_layer l = {0};
    l.layer_data = data;
    l.forward = zm_layer_forward_linear;
    l.backward = NULL;

    return l;
}

void zm_layer_forward_flatten(zm_layer *this, const zm_tensor *input) {
    if (input != this->input) {
        zm_tensor_destroy(this->output);
        u32 shape[] = {input->shape[0], input->_offs[0]};
        this->output = zm_tensor_create(2, shape, NULL);
    }
    memcpy(this->output.data, input->data, input->_size_flat * sizeof(f32));
}

zm_layer zm_layer_flatten() {
    zm_layer l = {0};
    l.forward = zm_layer_forward_flatten;
    l.backward = NULL;

    return l;
}

void zm_layer_forward_ReLU(zm_layer *this, const zm_tensor *input) {
    if (input != this->input) {
        zm_tensor_destroy(this->output);
        this->output = zm_tensor_create(input->dim, input->shape, NULL);
    }

    for (u32 i = 0; i < input->_size_flat; i ++)
        this->output.data[i] = input->data[i] * (input->data[i] > 0);
}

zm_layer zm_layer_ReLU() {
    zm_layer l = {0};
    l.forward = zm_layer_forward_ReLU;
    l.backward = NULL;

    return l;
}

typedef struct {
    u32 dim;
} zm_layer_data_softmax;

void zm_layer_forward_softmax(zm_layer *this, const zm_tensor *input) {
    zm_layer_data_softmax *ld = this->layer_data;
    if (input != this->input) {
        assert(ld->dim < input->dim);
        zm_tensor_destroy(this->output);
        this->output = zm_tensor_create(input->dim, input->shape, NULL);
    }

    zm_tensor *output = &this->output;

    u32 I = ld->dim ? input->_offs[ld->dim - 1] : input->_size_flat;
    u32 J = input->_offs[ld->dim];
    u32 K = input->shape[ld->dim];

    for (int i = 0; i < input->_size_flat; i += I) {
        f32 *oi = output->data + i;
        for (u32 j = 0; j < J; j ++) {
            f32 sum = 0;
            f32 *oj = oi + j;
            for (u32 k = 0; k < K; k ++)
                sum += (oj[k * J] = exp(input->data[i + j + k * J]));

            for (u32 k = 0; k < K; k ++)
                oj[k * J] /= sum;
        }
    }
}

zm_layer zm_layer_softmax(u32 dim) {
    zm_layer_data_softmax *data = malloc(sizeof(*data));
    data->dim = dim;

    zm_layer l = {0};
    l.layer_data = data;
    l.forward = zm_layer_forward_softmax;
    l.backward = NULL;

    return l;
}

zm_sequential zm_sequential_create(u32 n_layers, zm_layer *layers) {
    zm_sequential s = {0};
    s.n_layers = n_layers;
    s.layers = zm_copy(layers, n_layers * sizeof(*layers));
    return s;
}

const zm_tensor zm_sequential_forward(zm_sequential *s, const zm_tensor *input) {
    s->layers[0].forward(&s->layers[0], input);
    for (u32 i = 1; i < s->n_layers; i ++)
        s->layers[i].forward(&s->layers[i], &s->layers[i-1].output);
    return s->layers[s->n_layers - 1].output;
}
