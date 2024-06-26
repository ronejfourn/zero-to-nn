#include "../znn_common.h"

typedef struct {
    znn_tensor weights;
    znn_tensor biases;
} znn_layer_params_linear;

static ZNN_TENSOR_BACKWARD_FXN(_znn__backward_linear_n) {
    znn_tensor **prev = this->prev;
    u32 fi = prev[1]->shape[1];
    u32 fo = prev[1]->shape[0];
    u32 K = this->size / fo;

    #pragma omp parallel for
    for (u32 k = 0; k < K; k++) {
        f32 *iv = prev[0]->data + k * fi;
        f32 *og = this->grad + k * fo;
        f32 *wg = prev[1]->grad, *bg = prev[2]->grad;
        for (u32 i = 0; i < fo; i++, wg += fi) {
            bg[i] += og[i];
            for (u32 j = 0; j < fi; j ++)
                wg[j] += iv[j] * og[i];
        }
    }
}

static ZNN_TENSOR_BACKWARD_FXN(_znn__backward_linear_y) {
    znn_tensor **prev = this->prev;
    u32 fi = prev[1]->shape[1];
    u32 fo = prev[1]->shape[0];
    u32 K = this->size / fo;

    #pragma omp parallel for
    for (u32 k = 0; k < K; k ++) {
        f32 *iv = prev[0]->data + k * fi;
        f32 *ig = prev[0]->grad + k * fi;
        f32 *og = this->grad + k * fo;
        f32 *bg = prev[2]->grad;
        f32 *wv = prev[1]->data, *wg = prev[1]->grad;
        for (u32 i = 0; i < fo; i ++, wv += fi, wg += fi) {
            bg[i] += og[i];
            for (u32 j = 0; j < fi; j ++) {
                wg[j] += iv[j] * og[i];
                ig[j] += wv[j] * og[i];
            }
        }
    }
}

static ZNN_LAYER_INIT_FXN(_znn__init_linear) {
    znn_layer_params_linear *ld = this->parameters;
    znn_tensor *wt = &ld->weights;
    assert(input->shape[input->dim - 1] == wt->shape[1]);

    u32 *shape = znn_copy(input->shape, input->dim * 4);
    shape[input->dim - 1] = wt->shape[0];

    this->output = znn_tensor_from_shape(input->dim, shape);
    if (input->grad)
        ZNN_FXN_SET(this->output.backward, _znn__backward_linear_y);
    else
        ZNN_FXN_SET(this->output.backward, _znn__backward_linear_n);
    znn_tensor_require_grad(&this->output);
    znn_tensor_set_prev(&this->output, input, &ld->weights, &ld->biases);
}

static ZNN_LAYER_FORWARD_FXN(_znn__forward_linear) {
    znn_layer_params_linear *ld = this->parameters;
    znn_tensor *wt = &ld->weights;
    znn_tensor *bt = &ld->biases;
    znn_tensor *it = this->input;
    znn_tensor *ot = &this->output;

    u32 fi = wt->shape[1];
    u32 fo = wt->shape[0];
    u32 K = ot->size / fo;

    #pragma omp parallel for
    for (u32 k = 0; k < K; k ++) {
        f32 *ov = ot->data + k * fo;
        f32 *iv = it->data + k * fi;
        f32 *wv = wt->data, *bv = bt->data;
        for (u32 i = 0; i < fo; i ++, wv += fi) {
            ov[i] = bv[i];
            for (u32 j = 0; j < fi; j ++)
                ov[i] += wv[j] * iv[j];
        }
    }
}

static ZNN_LAYER_DESTROY_FXN(_znn__destroy_linear) {
    znn_layer_params_linear *ld = this->parameters;
    znn_tensor_destroy(ld->weights);
    znn_tensor_destroy(ld->biases);
    znn_free(ld);
}

znn_layer _znn_layer_linear(u32 in_features, u32 out_features, char *file, u32 line) {
    znn_trace(file, line);

    f32 k = sqrt(1.0/in_features);

    znn_layer_params_linear *params = znn_malloc(sizeof(*params));
    params->weights = znn_tensor_randr(
            -k, k, out_features, in_features);
    znn_tensor_require_grad(&params->weights);
    params->biases = znn_tensor_ones(out_features);
    znn_tensor_require_grad(&params->biases);

    znn_layer l = {0};
    l.n_params = 2;
    l.parameters = params;
    ZNN_FXN_SET(l.init, _znn__init_linear);
    ZNN_FXN_SET(l.forward, _znn__forward_linear);
    ZNN_FXN_SET(l.destroy, _znn__destroy_linear);

    return l;
}
