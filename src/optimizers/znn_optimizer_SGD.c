#include "../znn_optimizers.h"
#include <omp.h>

static ZNN_OPTIMIZER_STEP_FXN(_znn__step_SGD) {
#if ZNN_OPENMP_ENABLE
    #pragma omp parallel for
#endif
    for (u32 i = 0; i < this->n_params; i ++) {
        u32 S = this->parameters[i]->size;
        for (u32 j = 0; j < S; j ++)
            this->parameters[i]->data[j + 0] -=
                this->learning_rate * this->parameters[i]->grad[j + 0];
    }
}

znn_optimizer znn_optimizer_SGD(znn_tensor **params, u32 n_params, f32 learning_rate) {
    znn_optimizer s = {params, n_params, learning_rate};
    ZNN_FXN_SET(s.step, _znn__step_SGD);
    return s;
}
