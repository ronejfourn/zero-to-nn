#pragma once

#include "znn_tensor.h"

struct znn_layer;

#define ZNN_LAYER_FORWARD_FXN(F) \
    void F (struct znn_layer *this)
#define ZNN_LAYER_DESTROY_FXN(F) \
    void F (struct znn_layer *this)
#define ZNN_LAYER_INIT_FXN(F) \
    void F (struct znn_layer *this, znn_tensor *input)

typedef ZNN_LAYER_INIT_FXN((*znn_layer_init_fxn));
typedef ZNN_LAYER_FORWARD_FXN((*znn_layer_forward_fxn));
typedef ZNN_LAYER_DESTROY_FXN((*znn_layer_destroy_fxn));

typedef struct znn_layer {
    znn_tensor *input;
    u32 n_params;
    void *parameters;
    znn_tensor output;
    ZNN_FXN(znn_layer_init_fxn) init;
    ZNN_FXN(znn_layer_forward_fxn) forward;
    ZNN_FXN(znn_layer_destroy_fxn) destroy;
} znn_layer;

typedef struct znn_sequential {
    znn_layer *layers;
    u32 n_layers;
    znn_tensor **parameters;
    u32 n_params;
} znn_sequential;

znn_tensor *znn_layer_forward(znn_layer *this, znn_tensor *input);

#define znn_layer_flatten() _znn_layer_flatten(__FILE__, __LINE__)
#define znn_layer_linear(...) _znn_layer_linear(__VA_ARGS__, __FILE__, __LINE__)
#define znn_layer_ReLU() _znn_layer_ReLU(__FILE__, __LINE__)
#define znn_layer_softmax(...) _znn_layer_softmax(__VA_ARGS__, __FILE__, __LINE__)

znn_layer _znn_layer_flatten(char *file, u32 line);
znn_layer _znn_layer_linear(u32 in_features, u32 out_features, char *file, u32 line);
znn_layer _znn_layer_ReLU(char *file, u32 line);
znn_layer _znn_layer_softmax(u32 dim, char *file, u32 line);

#define znn_layer_destroy(...) _znn_layer_destroy(__VA_ARGS__, __FILE__, __LINE__)
void _znn_layer_destroy(struct znn_layer this, char *file, u32 line);

znn_tensor *znn_sequential_forward(znn_sequential *s, znn_tensor *input);

#define znn_sequential_create(...) _znn_sequential_create(__VA_ARGS__, __FILE__, __LINE__)
znn_sequential _znn_sequential_create(znn_layer *layers, u32 count, char *file, u32 line);

#define znn_sequential_destroy(...) _znn_sequential_destroy(__VA_ARGS__, __FILE__, __LINE__)
void _znn_sequential_destroy(znn_sequential s, char *file, u32 line);
