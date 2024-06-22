#pragma once

#include "zm_tensor.h"

struct zm_layer;

#define ZM_LAYER_FORWARD_FXN(F) void F (struct zm_layer *this, zm_tensor *input)
#define ZM_LAYER_BACKWARD_FXN(F) void F (struct zm_layer *this)
#define ZM_LAYER_DESTROY_FXN(F) void F (struct zm_layer this)

typedef ZM_LAYER_FORWARD_FXN((*zm_layer_forward_fxn));
typedef ZM_LAYER_BACKWARD_FXN((*zm_layer_backward_fxn));
typedef ZM_LAYER_DESTROY_FXN((*zm_layer_destroy_fxn));

typedef struct zm_layer {
    zm_tensor *input;
    zm_tensor output;
    zm_layer_forward_fxn forward;
    zm_layer_backward_fxn backward;
    zm_layer_destroy_fxn destroy;
    void *layer_data;
} zm_layer;

typedef struct zm_sequential {
    zm_layer *layers;
    u32 n_layers;
} zm_sequential;

#define zm_layer_flatten() _zm_layer_flatten(__FILE__, __LINE__)
#define zm_layer_linear(...) _zm_layer_linear(__VA_ARGS__, __FILE__, __LINE__)
#define zm_layer_ReLU() _zm_layer_ReLU(__FILE__, __LINE__)
#define zm_layer_softmax(...) _zm_layer_softmax(__VA_ARGS__, __FILE__, __LINE__)

zm_layer _zm_layer_flatten(char *file, u32 line);
zm_layer _zm_layer_linear(u32 in_features, u32 out_features, char *file, u32 line);
zm_layer _zm_layer_ReLU(char *file, u32 line);
zm_layer _zm_layer_softmax(u32 dim, char *file, u32 line);

const zm_tensor *zm_sequential_forward(zm_sequential *s, zm_tensor *input);
void zm_sequential_backward(zm_sequential *s);

void zm_sequential_destroy(zm_sequential s);
