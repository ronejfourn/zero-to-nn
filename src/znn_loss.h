#pragma once

#include "znn_tensor.h"

struct znn_loss;

#define ZNN_LOSS_FXN(F) \
    void F (struct znn_loss *this, znn_tensor *input, znn_tensor *target)

#define ZNN_LOSS_DESTROY_FXN(F) \
    void F (struct znn_loss this)

typedef ZNN_LOSS_FXN((*znn_loss_fxn));
typedef ZNN_LOSS_DESTROY_FXN((*znn_loss_destroy_fxn));

typedef struct znn_loss {
    znn_tensor *input;
    znn_loss_fxn fn;
    znn_loss_destroy_fxn destroy;
    znn_tensor output;
} znn_loss;

znn_tensor *znn_loss_calc(znn_loss *this, znn_tensor *input, znn_tensor *target);

#define znn_loss_mse() _znn_loss_mse(__FILE__, __LINE__)
znn_loss _znn_loss_mse(char *file, u32 line);

#define znn_loss_destroy(...) _znn_loss_destroy(__VA_ARGS__, __FILE__, __LINE__)
void _znn_loss_destroy(znn_loss this, char *file, u32 line);
