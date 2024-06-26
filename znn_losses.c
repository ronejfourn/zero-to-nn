#include "znn_losses.h"

#include "znn_losses/znn_MSE.c"

znn_tensor *znn_loss_calc(znn_loss *this, znn_tensor *input, znn_tensor *target) {
    this->calc.fn(this, input, target);
    return &this->output;
}

void _znn_loss_destroy(znn_loss this, char *file, u32 line) {
    znn_trace(file, line);
    if (this.destroy.fn) this.destroy.fn(&this);
    znn_tensor_destroy(this.output);
}
