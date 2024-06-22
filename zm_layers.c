#include "zm_util.h"
#include "zm_layers.h"

#include <assert.h>
#include <string.h>
#include <math.h>

const zm_tensor *zm_sequential_forward(zm_sequential *s, zm_tensor *input) {
    s->layers[0].forward(&s->layers[0], input);
    for (u32 i = 1; i < s->n_layers; i ++)
        s->layers[i].forward(&s->layers[i], &s->layers[i-1].output);
    return &s->layers[s->n_layers - 1].output;
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
