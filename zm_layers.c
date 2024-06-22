#include "zm_util.h"
#include "zm_layers.h"

#include <string.h>
#include <math.h>

zm_tensor *zm_layer_forward(struct zm_layer *this, zm_tensor *input) {
    this->forward(this, input);
    return &this->output;
}

zm_tensor *zm_sequential_forward(zm_sequential *s, zm_tensor *input) {
    for (u32 i = 1; i < s->n_layers; i ++)
        input = zm_layer_forward(s->layers + i, input);
    return input;
}

/* void zm_sequential_backward(zm_sequential *s) { */
/*     for (u32 i = 0; i < s->n_layers; i ++) */
/*         s->layers[s->n_layers - i - 1].backward(s->layers + s->n_layers - i - 1); */
/* } */

void zm_sequential_destroy(zm_sequential s) {
    for (u32 i = 0; i < s.n_layers; i ++)
        s.layers[i].destroy(s.layers[i]);
    s.n_layers = 0;
    s.layers = NULL;
}
