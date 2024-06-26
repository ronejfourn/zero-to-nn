#pragma once

#include <stdio.h>
#include "znn_tensor.h"

typedef struct {
    void *fptr;
    u32 base;
    u32 dim;
    u32 *shape;
    u32 size;
    u32 end;
    u8 type;
} znn_dataset_idx;

typedef struct znn_dataset {
    znn_dataset_idx data;
    znn_dataset_idx label;
} znn_dataset;

#define znn_dataset_load_idx(D, L) _znn_dataset_load_idx(D, L, __FILE__, __LINE__)
znn_dataset _znn_dataset_load_idx(const char *dpath, const char *lpath, const char *file, u32 line);
#define znn_dataset_destroy(D) _znn_dataset_destroy(D, __FILE__, __LINE__)
void _znn_dataset_destroy(znn_dataset d, const char *file, u32 line);
#define znn_dataset_get_batch(D, B, X, Y) _znn_dataset_get_batch(D, B, X, Y, __FILE__, __LINE__)
bool _znn_dataset_get_batch(znn_dataset *d, u32 batch_size, znn_tensor *x, znn_tensor *y, const char *file, u32 line);
