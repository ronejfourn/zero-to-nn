#include "znn_layers.h"

#include "znn_layers/znn_ReLU.c"
#include "znn_layers/znn_flatten.c"
#include "znn_layers/znn_linear.c"
#include "znn_layers/znn_softmax.c"

// TODO: currently assumes input shape never changes
znn_tensor *znn_layer_forward(struct znn_layer *this, znn_tensor *input) {
    if (!this->input) {
        this->input = input;
        znn_tensor_destroy(this->output);
        this->init.fn(this, input);
    }

    this->forward.fn(this);
    return &this->output;
}

void _znn_layer_destroy(struct znn_layer this, char *file, u32 line) {
    znn_trace(file, line);
    if (this.destroy.fn) this.destroy.fn(&this);
    znn_tensor_destroy(this.output);
}

znn_sequential _znn_sequential_create(znn_layer *layers, u32 count, char *file, u32 line) {
    znn_trace(file, line);
    znn_sequential s = {layers, count, 0, 0};
    for (u32 i = 0; i < s.n_layers; i ++)
        s.n_params += layers[i].n_params;
    s.parameters = znn_malloc(s.n_params * sizeof(*s.parameters));
    u32 k = 0;
    for (u32 i = 0; i < s.n_layers; i ++)
        for (u32 j = 0; j < layers[i].n_params; j ++)
            s.parameters[k++] = (znn_tensor*)(layers[i].parameters) + j;
    return s;
}

znn_tensor *znn_sequential_forward(znn_sequential *s, znn_tensor *input) {
    for (u32 i = 0; i < s->n_layers; i ++)
        input = znn_layer_forward(s->layers + i, input);

    return input;
}

void _znn_sequential_destroy(znn_sequential s, char *file, u32 line) {
    znn_trace(file, line);
    for (u32 i = 0; i < s.n_layers; i ++)
        znn_layer_destroy(s.layers[i]);
}
