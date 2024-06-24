#include "../znn_loss.h"
#include "../znn_util.h"

ZNN_TENSOR_BACKWARD_FXN(znn_loss_backward_mse) {
    znn_tensor **prev = this->prev;
    znn_tensor *input = prev[0];
    znn_tensor *target = prev[1];
    if (!input->grad) return;

    u32 N = input->step[0];
    for (u32 i = 0; i < input->shape[0]; i ++) {
        f32 *x = input->data + i * N;
        f32 *y = target->data + i * N;
        f32 *xg = input->grad + i * N;
        for (u32 j = 0; j < N; j ++)
            xg[j] = 2 * (x[j] - y[j]) / (N * input->shape[0]);
    }
}

ZNN_LOSS_FXN(znn_loss_fxn_mse) {
    if (this->input != input) {
        this->input = input;
        znn_tensor_set_prev(&this->output, input, target);
    }

    assert(target->dim == input->dim);
    for (u32 i = 0; i < target->dim; i ++)
        assert(target->shape[i] == input->shape[i]);

    u32 N = input->step[0];
    this->output.data[0] = 0;
    for (u32 i = 0; i < input->shape[0]; i ++) {
        f32 *x = input->data + i * N;
        f32 *y = target->data + i * N;
        f32 e = 0;
        for (u32 j = 0; j < N; j ++)
            e += (x[j] - y[j]) * (x[j] - y[j]);
        this->output.data[0] += e / (N * input->shape[0]);
    }
}

ZNN_LOSS_DESTROY_FXN(znn_loss_destroy_mse) {
    znn_tensor_destroy(this.output);
}

znn_loss _znn_loss_mse(char *file, u32 line) {
    znn_trace(file, line);
    znn_loss l = {0};
    l.fn = znn_loss_fxn_mse;
    l.destroy = znn_loss_destroy_mse;

    l.output = znn_tensor_create(1);
    l.output.backward = znn_loss_backward_mse;

    return l;
}
