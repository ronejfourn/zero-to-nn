#include "znn_util.h"
#include "znn_optimizers.h"

void znn_optimizer_zero_grad(znn_optimizer *o) {
    for (u32 i = 0; i < o->n_params; i ++)
        memset(o->parameters[i]->grad, 0, o->parameters[i]->size * 4);
}

void znn_optimizer_step(znn_optimizer *o) {
    o->step.fn(o);
}
