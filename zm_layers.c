#include "zm_util.h"
#include "zm_layers.h"

// TODO: currently assumes input shape never changes
zm_tensor *zm_layer_forward(struct zm_layer *this, zm_tensor *input) {
    if (!this->input) {
        this->input = input;
        zm_tensor_destroy(this->output);
        this->init(this, input);
    }

    this->forward(this);
    return &this->output;
}

zm_tensor *zm_sequential_forward(zm_sequential *s, zm_tensor *input) {
    for (u32 i = 0; i < s->n_layers; i ++)
        input = zm_layer_forward(s->layers + i, input);

    return input;
}

void _zm_layer_destroy(struct zm_layer this, char *file, u32 line) {
    zm_trace(file, line);
    if (this.destroy) this.destroy(&this);
    zm_tensor_destroy(this.output);
}

zm_sequential _zm_sequential_create(zm_layer *layers, u32 count, char *file, u32 line) {
    zm_trace(file, line);
    zm_sequential s = {layers, count, 0, 0};
    for (u32 i = 0; i < s.n_layers; i ++)
        s.n_params += layers[i].n_params;
    s.parameters = zm_malloc(s.n_params * sizeof(*s.parameters));
    u32 k = 0;
    for (u32 i = 0; i < s.n_layers; i ++)
        for (u32 j = 0; j < layers[i].n_params; j ++)
            s.parameters[k++] = (zm_tensor*)(layers[i].parameters) + j;
    return s;
}

void _zm_sequential_destroy(zm_sequential s, char *file, u32 line) {
    zm_trace(file, line);
    for (u32 i = 0; i < s.n_layers; i ++)
        zm_layer_destroy(s.layers[i]);
}
