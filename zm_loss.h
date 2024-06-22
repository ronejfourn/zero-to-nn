#pragma once

#include "zm_tensor.h"

struct zm_loss;

#define ZM_LOSS_FXN(F) \
    void F (struct zm_loss *this, zm_tensor *input, zm_tensor *target)

typedef ZM_LOSS_FXN((*zm_loss_fxn));

typedef struct zm_loss {
    zm_tensor *input;
    zm_loss_fxn fn;
    zm_tensor output;
} zm_loss;

zm_tensor *zm_loss_calc(zm_loss *this, zm_tensor *input, zm_tensor *target);

zm_loss zm_loss_mse();
