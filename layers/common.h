#pragma once
#include "../zm_util.h"
#include "../zm_layers.h"
#include <math.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>

#define L(_L) \
    zm_layer l = {0}; \
    l.forward = zm_layer_forward_ ## _L; \
    l.destroy = zm_layer_destroy_ ## _L
