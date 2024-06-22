#include "zm_util.h"
#include "zm_layers.h"

#include <string.h>
#include <math.h>

zm_tensor *zm_layer_forward(struct zm_layer *this, zm_tensor *input) {
    this->forward(this, input);
    return &this->output;
}

zm_tensor *zm_sequential_forward(zm_sequential *s, zm_tensor *input) {
    for (u32 i = 0; i < s->n_layers; i ++)
        input = zm_layer_forward(s->layers + i, input);
    return input;
}

void _zm_layer_destroy(struct zm_layer this, char *file, u32 line) {
    zm_trace(file, line);
    this.destroy(this);
}

void zm_layer_shd(zm_layer *this, f32 lr) {
    if (this->shd) this->shd(this, lr);
}

void zm_layer_zero_grad(zm_layer *this) {
    if (this->zero_grad) this->zero_grad(this);
    memset(this->output.grad, 0, this->output.size * 4);
}

void _zm_sequential_destroy(zm_sequential s, char *file, u32 line) {
    zm_trace(file, line);
    for (u32 i = 0; i < s.n_layers; i ++)
        zm_layer_destroy(s.layers[i]);
    s.n_layers = 0;
    s.layers = NULL;
}

void zm_sequential_shd(zm_sequential *s, f32 lr) {
    for (u32 i = 0; i < s->n_layers; i ++)
        zm_layer_shd(s->layers + i, lr);
}

void zm_sequential_zero_grad(zm_sequential *s) {
    for (u32 i = 0; i < s->n_layers; i ++)
        zm_layer_zero_grad(s->layers + i);
}
