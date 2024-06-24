#pragma once

#include "zm_tensor.h"

struct zm_optimizer;

#define ZM_OPTIMIZER_STEP_FXN(F) \
    void F (struct zm_optimizer *this)

typedef ZM_OPTIMIZER_STEP_FXN((*zm_optimizer_step_fxn));

typedef struct zm_optimizer {
    zm_tensor **parameters;
    u32 n_params;
    f32 learning_rate;
    zm_optimizer_step_fxn step;
} zm_optimizer;

zm_optimizer zm_optimizer_SGD(zm_tensor **params, u32 n_params, f32 learning_rate);
void zm_optimizer_step(zm_optimizer *this);
void zm_optimizer_zero_grad(zm_optimizer *this);
