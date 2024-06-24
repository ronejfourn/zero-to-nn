#include "znn_util.h"
#include "znn_loss.h"

znn_tensor *znn_loss_calc(znn_loss *this, znn_tensor *input, znn_tensor *target) {
    this->fn(this, input, target);
    return &this->output;
}

void _znn_loss_destroy(znn_loss this, char *file, u32 line) {
    znn_trace(file, line);
    this.destroy(this);
}
