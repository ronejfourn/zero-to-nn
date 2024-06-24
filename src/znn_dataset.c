#include "znn_util.h"
#include "znn_dataset.h"

#define FREAD(...) assert(fread(__VA_ARGS__))

enum {
    ZNN_IDX_UBYTE  = 0x08,
    ZNN_IDX_BYTE   = 0x09,
    ZNN_IDX_SHORT  = 0x0B,
    ZNN_IDX_INT    = 0x0C,
    ZNN_IDX_FLOAT  = 0x0D,
    ZNN_IDX_DOUBLE = 0x0E,
};

static inline u32 idx_sizeof(u8 t) {
    switch (t) {
    case ZNN_IDX_UBYTE  : return 1;
    case ZNN_IDX_BYTE   : return 1;
    case ZNN_IDX_SHORT  : return 2;
    case ZNN_IDX_INT    : return 4;
    case ZNN_IDX_FLOAT  : return 4;
    case ZNN_IDX_DOUBLE : return 8;
    default: znn_unreachable();
    }
}

static inline u32 idx_read_u32(FILE *f) {
    u32 h;
    FREAD(&h, 4, 1, f);
    return znn_correct_endian32(h);
}

static inline u32 idx_read_header(FILE *f, u8 *t) {
    u32 h = idx_read_u32(f);
    assert((h & 0xffff0000) == 0x0);

    *t = (h & 0xff00) >> 8;
    assert(*t == 8 || *t == 9 || (*t > 0xA && *t < 0xF));

    return h & 0xff;
}

znn_dataset_idx _znn_dataset_load_idx_file(const char *path) {
    FILE *fptr = fopen(path, "rb");
    znn_dataset_idx d = {0};
    d.dim = idx_read_header(fptr, &d.type);
    d.shape = znn_malloc(d.dim * 4);

    d.size = 1;
    for (u32 i = 0; i < d.dim; i ++) {
        d.shape[i] = idx_read_u32(fptr);
        d.size *= d.shape[i];
    }

    d.size /= d.shape[0];
    d.base = ftell(fptr);
    d.fptr = fptr;

    fseek(fptr, 0, SEEK_END);
    d.end = ftell(fptr) - d.base;
    fseek(fptr, d.base, SEEK_SET);
    return d;
}

znn_dataset znn_dataset_load_idx(const char *dpath, const char *lpath) {
    znn_dataset d = {0};
    d.data = _znn_dataset_load_idx_file(dpath);
    d.label = _znn_dataset_load_idx_file(lpath);
    assert(d.data.shape[0] == d.label.shape[0]);
    return d;
}

void _znn_dataset_destroy_idx(znn_dataset_idx d) {
    znn_free(d.shape);
    fclose(d.fptr);
}

void znn_dataset_destroy(znn_dataset d) {
    _znn_dataset_destroy_idx(d.data);
    _znn_dataset_destroy_idx(d.label);
}

bool _znn_dataset_get_batch_idx(znn_dataset_idx *d, u32 bs, znn_tensor *x) {
    u32 S = d->size * idx_sizeof(d->type);

    if (ftell(d->fptr) + bs * S > d->end) {
        fseek(d->fptr, d->base, SEEK_SET);
        return false;
    }

    u32 *shape = znn_copy(d->shape, d->dim * 4);
    u32 L = S - ((S - 1) & ~7);

    shape[0] = bs;
    *x = znn_tensor_create_from_shape(d->dim, shape);

    // TODO: proper support for all data types
    for (u32 i = 0; i < bs; i ++) {
        f32 *xp = x->data + i * S;
        for (u32 j = 0, l = L; j < S; j += l, l = 8) {
            u8 buf[8] = {0};
            FREAD(buf, L, 1, d->fptr);
            switch (d->type) {
            case ZNN_IDX_UBYTE  : {
                for (u32 k = 0; k < L; k ++)
                    *(xp++) = buf[k];
            } break;
            case ZNN_IDX_BYTE   : {
                for (u32 k = 0; k < L; k ++)
                    *(xp++) = *(i8*)(buf + k);
            } break;
            case ZNN_IDX_SHORT  : {
                for (u32 k = 0; k < L; k += 2) {
                    u16 a = znn_correct_endian16(*(u16*)(buf + k));
                    *(xp++) = *(i16*)&a;
                }
            } break;
            case ZNN_IDX_INT    : {
                for (u32 k = 0; k < L; k += 4) {
                    u32 a = znn_correct_endian32(*(u32*)(buf + k));
                    *(xp++) = *(i32*)&a;
                }
            } break;
            case ZNN_IDX_FLOAT  : {
                for (u32 k = 0; k < L; k += 4) {
                    u32 a = znn_correct_endian32(*(u32*)(buf + k));
                    *(xp++) = *(f32*)&a;
                }
            } break;
            case ZNN_IDX_DOUBLE : {
                u64 a = znn_correct_endian64(*(u64*)buf);
                *(xp++) = *(f64*)&a;
            } break;
            default: znn_unreachable();
            }
        }
    }
}

bool znn_dataset_get_batch(znn_dataset *d, u32 bs, znn_tensor *x, znn_tensor *y) {
    bool a = _znn_dataset_get_batch_idx(&d->data, bs, x);
    bool b = _znn_dataset_get_batch_idx(&d->label, bs, y);
    assert(a == b);
    return a;
}
