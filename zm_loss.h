#pragma once

#include "zm_tensor.h"

struct zm_loss;

#define ZM_LOSS_FXN(F) \
    void F (struct zm_loss *this, zm_tensor *input, zm_tensor *target)

#define ZM_LOSS_DESTROY_FXN(F) \
    void F (struct zm_loss this)

typedef ZM_LOSS_FXN((*zm_loss_fxn));
typedef ZM_LOSS_DESTROY_FXN((*zm_loss_destroy_fxn));

typedef struct zm_loss {
    zm_tensor *input;
    zm_loss_fxn fn;
    zm_loss_destroy_fxn destroy;
    zm_tensor output;
} zm_loss;

zm_tensor *zm_loss_calc(zm_loss *this, zm_tensor *input, zm_tensor *target);

#define zm_loss_mse() _zm_loss_mse(__FILE__, __LINE__)
zm_loss _zm_loss_mse(char *file, u32 line);

#define zm_loss_destroy(...) _zm_loss_destroy(__VA_ARGS__, __FILE__, __LINE__)
void _zm_loss_destroy(zm_loss this, char *file, u32 line);
