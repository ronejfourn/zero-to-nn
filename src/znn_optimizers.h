#pragma once

#include "znn_tensor.h"

struct znn_optimizer;

#define ZNN_OPTIMIZER_STEP_FXN(F) \
    void F (struct znn_optimizer *this)

typedef ZNN_OPTIMIZER_STEP_FXN((*znn_optimizer_step_fxn));

typedef struct znn_optimizer {
    znn_tensor **parameters;
    u32 n_params;
    f32 learning_rate;
    znn_optimizer_step_fxn step;
} znn_optimizer;

znn_optimizer znn_optimizer_SGD(znn_tensor **params, u32 n_params, f32 learning_rate);
void znn_optimizer_step(znn_optimizer *this);
void znn_optimizer_zero_grad(znn_optimizer *this);
