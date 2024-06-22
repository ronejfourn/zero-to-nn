#include "zm_util.h"
#include "zm_loss.h"

zm_tensor *zm_loss_calc(zm_loss *this, zm_tensor *input, zm_tensor *target) {
    this->fn(this, input, target);
    return &this->output;
}

void _zm_loss_destroy(zm_loss this, char *file, u32 line) {
    zm_trace(file, line);
    this.destroy(this);
}
