#include "zm_util.h"
#include "zm_loss.h"

#include <assert.h>

zm_tensor *zm_loss_calc(zm_loss *this, zm_tensor *input, zm_tensor *target) {
    this->fn(this, input, target);
    return &this->output;
}

ZM_TENSOR_BACKWARD_FXN(zm_loss_backward_mse) {
    zm_tensor **prev = this->prev;
    zm_tensor *input = prev[0];
    zm_tensor *target = prev[1];
    if (!input->grad) return;

    u32 N = input->_offs[0];
    for (u32 i = 0; i < input->shape[0]; i ++) {
        f32 *x = input->data + i * N;
        f32 *xg = input->grad + i * N;
        f32 *tg = this->grad + i * N;
        f32 *y = target->data + i * N;
        for (u32 j = 0; j < N; j ++)
            xg[j] += 2 * (x[j] - y[j]);
    }
}

ZM_LOSS_FXN(zm_loss_fxn_mse) {
    if (this->input != input) { //TODO: assert same shape
        this->input = input;
        zm_tensor_destroy(this->output);
        u32 shape[] = {1, input->shape[0]};
        this->output = zm_tensor_create(2, shape, NULL, false);
        this->output.backward = zm_loss_backward_mse;
        zm_tensor *prev[] = {input, target};
        zm_tensor_set_prev(&this->output, prev, 2);
    }

    u32 N = input->_offs[0];
    for (u32 i = 0; i < input->shape[0]; i ++) {
        f32 *x = input->data + i * N;
        f32 *y = target->data + i * N;
        f32 e = 0;
        for (u32 j = 0; j < N; j ++)
            e += (x[j] - y[j]) * (x[j] - y[j]);
        this->output.data[i] = e;
    }
}

zm_loss zm_loss_mse() {
    zm_loss l = {0};
    l.fn = zm_loss_fxn_mse;

    return l;
}
