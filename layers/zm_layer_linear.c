#include <omp.h>
#include "common.h"

typedef struct {
    zm_tensor weights;
    zm_tensor biases;
} zm_layer_params_linear;

ZM_TENSOR_BACKWARD_FXN(zm_layer_backward_linear_n) {
    zm_tensor **prev = this->prev;
    u32 fi = prev[1]->shape[1];
    u32 fo = prev[1]->shape[0];

    u32 K = this->size / fo;
    u32 L = fi - (fi & ~3);

    #pragma omp parallel for
    for (u32 k = 0; k < K; k++) {
        f32 *iv = prev[0]->data + k * fi;
        f32 *og = this->grad + k * fo;
        f32 *wg = prev[1]->grad, *bg = prev[2]->grad;
        for (u32 i = 0; i < fo; i++, wg += fi) {
            bg[i] += og[i];
            for (u32 j = 0; j < L; j ++)
                wg[j + 0] += iv[j + 0] * og[i];
            for (u32 j = L; j < fi; j += 4) {
                wg[j + 0] += iv[j + 0] * og[i];
                wg[j + 1] += iv[j + 1] * og[i];
                wg[j + 2] += iv[j + 2] * og[i];
                wg[j + 3] += iv[j + 3] * og[i];
            }
        }
    }
}

ZM_TENSOR_BACKWARD_FXN(zm_layer_backward_linear_y) {
    zm_tensor **prev = this->prev;
    u32 fi = prev[1]->shape[1];
    u32 fo = prev[1]->shape[0];
    
    u32 K = this->size / fo;
    u32 L = fi - (fi & ~3);
    
    #pragma omp parallel for
    for (u32 k = 0; k < K; k ++) {
        f32 *iv = prev[0]->data + k * fi;
        f32 *ig = prev[0]->grad + k * fi;
        f32 *og = this->grad + k * fo;
        f32 *bg = prev[2]->grad;
        f32 *wv = prev[1]->data, *wg = prev[1]->grad;
        for (u32 i = 0; i < fo; i ++, wv += fi, wg += fi) {
            bg[i] += og[i];
            for (u32 j = 0; j < L; j ++) {
                wg[j + 0] += iv[j + 0] * og[i];
                ig[j + 0] += wv[j + 0] * og[i];
            }
            for (u32 j = L; j < fi; j += 4) {
                wg[j + 0] += iv[j + 0] * og[i];
                wg[j + 1] += iv[j + 1] * og[i];
                wg[j + 2] += iv[j + 2] * og[i];
                wg[j + 3] += iv[j + 3] * og[i];
                ig[j + 0] += wv[j + 0] * og[i];
                ig[j + 1] += wv[j + 1] * og[i];
                ig[j + 2] += wv[j + 2] * og[i];
                ig[j + 3] += wv[j + 3] * og[i];
            }
        }
    }
}

ZM_LAYER_INIT_FXN(zm_layer_init_linear) {
    zm_layer_params_linear *ld = this->parameters;
    zm_tensor *wt = &ld->weights;
    assert(input->shape[input->dim - 1] == wt->shape[1]);

    u32 *shape = zm_copy(input->shape, input->dim * 4);
    shape[input->dim - 1] = wt->shape[0];

    this->output = zm_tensor_create_from_shape(input->dim, shape);
    this->output.backward = input->grad ? 
        zm_layer_backward_linear_y : zm_layer_backward_linear_n;
    zm_tensor_require_grad(&this->output);
    zm_tensor_set_prev(&this->output, input, &ld->weights, &ld->biases);
}

ZM_LAYER_FORWARD_FXN(zm_layer_forward_linear) {
    zm_layer_params_linear *ld = this->parameters;
    zm_tensor *wt = &ld->weights;
    zm_tensor *bt = &ld->biases;
    zm_tensor *it = this->input;
    zm_tensor *ot = &this->output;
    
    u32 fi = wt->shape[1];
    u32 fo = wt->shape[0];
    
    u32 K = ot->size / fo;
    u32 L = fi - (fi & ~3);
    
    #pragma omp parallel for
    for (u32 k = 0; k < K; k ++) {
        f32 *ov = ot->data + k * fo;
        f32 *iv = it->data + k * fi;
        f32 *wv = wt->data, *bv = bt->data;
        for (u32 i = 0; i < fo; i ++, wv += fi) {
            ov[i] = bv[i];
            for (u32 j = 0; j < L; j ++)
                ov[i] += wv[j + 0] * iv[j + 0];
            for (u32 j = L; j < fi; j += 4) {
                ov[i] += wv[j + 0] * iv[j + 0];
                ov[i] += wv[j + 1] * iv[j + 1];
                ov[i] += wv[j + 2] * iv[j + 2];
                ov[i] += wv[j + 3] * iv[j + 3];
            }
        }
    }
}

ZM_LAYER_DESTROY_FXN(zm_layer_destroy_linear) {
    zm_layer_params_linear *ld = this->parameters;
    zm_tensor_destroy(ld->weights);
    zm_tensor_destroy(ld->biases);
    zm_free(ld);
}

zm_layer _zm_layer_linear(u32 in_features, u32 out_features, char *file, u32 line) {
    zm_trace(file, line);

    f32 k = sqrt(1.0/in_features);

    zm_layer_params_linear *params = zm_malloc(sizeof(*params));
    params->weights = zm_tensor_randr(
            -k, k, out_features, in_features);
    zm_tensor_require_grad(&params->weights);
    params->biases = zm_tensor_ones(out_features);
    zm_tensor_require_grad(&params->biases);

    zm_layer l = {0};
    l.n_params = 2;
    l.init = zm_layer_init_linear;
    l.forward = zm_layer_forward_linear;
    l.destroy = zm_layer_destroy_linear;
    l.parameters = params;

    return l;
}
