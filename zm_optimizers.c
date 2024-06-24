#include "zm_util.h"
#include "zm_optimizers.h"

void zm_optimizer_zero_grad(zm_optimizer *o) {
    for (u32 i = 0; i < o->n_params; i ++)
        memset(o->parameters[i]->grad, 0, o->parameters[i]->size * 4);
}

void zm_optimizer_step(zm_optimizer *o) {
    o->step(o);
}
