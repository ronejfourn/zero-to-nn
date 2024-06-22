#include "../zm_loss.h"
#include "../zm_util.h"

ZM_TENSOR_BACKWARD_FXN(zm_loss_backward_mse) {
    zm_tensor **prev = this->prev;
    zm_tensor *input = prev[0];
    zm_tensor *target = prev[1];
    if (!input->grad) return;

    u32 N = input->_offs[0];
    for (u32 i = 0; i < input->shape[0]; i ++) {
        f32 *x = input->data + i * N;
        f32 *y = target->data + i * N;
        f32 *xg = input->grad + i * N;
        for (u32 j = 0; j < N; j ++)
            xg[j] = 2 * (x[j] - y[j]);
    }
}

ZM_LOSS_FXN(zm_loss_fxn_mse) {
    if (this->input != input) { //TODO: assert same shape
        this->input = input;
        zm_tensor *prev[] = {input, target};
        zm_tensor_set_prev(&this->output, prev, 2);
    }

    u32 N = input->_offs[0];
    this->output.data[0] = 0;
    for (u32 i = 0; i < input->shape[0]; i ++) {
        f32 *x = input->data + i * N;
        f32 *y = target->data + i * N;
        f32 e = 0;
        for (u32 j = 0; j < N; j ++)
            e += (x[j] - y[j]) * (x[j] - y[j]);
        this->output.data[0] += e;
    }
}

ZM_LOSS_DESTROY_FXN(zm_loss_destroy_mse) {
    zm_tensor_destroy(this.output);
}

zm_loss _zm_loss_mse(char *file, u32 line) {
    zm_trace(file, line);
    zm_loss l = {0};
    l.fn = zm_loss_fxn_mse;
    l.destroy = zm_loss_destroy_mse;

    u32 shape[] = {1};
    l.output = zm_tensor_create(1, shape, NULL, false);
    l.output.backward = zm_loss_backward_mse;

    return l;
}
