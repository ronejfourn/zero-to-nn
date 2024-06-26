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

static inline u32 _znn__idx_sizeof(u8 t) {
    switch (t) {
    case ZNN_IDX_UBYTE  : return 1;
    case ZNN_IDX_BYTE   : return 1;
    case ZNN_IDX_SHORT  : return 2;
    case ZNN_IDX_INT    : return 4;
    case ZNN_IDX_FLOAT  : return 4;
    case ZNN_IDX_DOUBLE : return 8;
    default: znn_unreachable();
    }
    return 0;
}

static inline u32 _znn__idx_read_u32(FILE *f) {
    u32 h;
    FREAD(&h, 4, 1, f);
    return znn_correct_endian32(h);
}

static inline u32 _znn__idx_read_hdr(FILE *f, u8 *t) {
    u32 h = _znn__idx_read_u32(f);
    assert((h & 0xffff0000) == 0x0);

    *t = (h & 0xff00) >> 8;
    assert(*t == 8 || *t == 9 || (*t > 0xA && *t < 0xF));

    return h & 0xff;
}

static znn_dataset_idx _znn__idx_load_file(const char *path) {
    FILE *fptr = fopen(path, "rb");
    if (!fptr) {
        fprintf(stderr, "fopen(\"%s\", \"rb\" failed)\n", path);
        assert(fptr);
    }

    znn_dataset_idx d = {0};
    d.dim = _znn__idx_read_hdr(fptr, &d.type);
    d.shape = znn_malloc(d.dim * 4);

    d.size = 1;
    for (u32 i = 0; i < d.dim; i ++) {
        d.shape[i] = _znn__idx_read_u32(fptr);
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

static void _znn__idx_destroy(znn_dataset_idx d) {
    znn_free(d.shape);
    fclose(d.fptr);
}

static bool _znn__idx_get_batch(znn_dataset_idx *d, u32 bs, znn_tensor *x) {
    u32 S = d->size * _znn__idx_sizeof(d->type);

    if (ftell(d->fptr) + bs * S > d->end) {
        fseek(d->fptr, d->base, SEEK_SET);
        return false;
    }

    u32 *shape = znn_copy(d->shape, d->dim * 4);
    u32 L = S - ((S - 1) & ~7);

    shape[0] = bs;
    *x = znn_tensor_from_shape(d->dim, shape);

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
                for (u32 k = 0; k < (L >> 2); k ++) {
                    f32 *fbuf = (f32*)buf;
                    u32 *ubuf = (u32*)buf;
                    ubuf[k] = znn_correct_endian32(*(u32*)(buf + (k << 2)));
                    *(xp++) = fbuf[k];
                }
            } break;
            case ZNN_IDX_DOUBLE : {
                f64 *fbuf = (f64*)buf;
                u64 *ubuf = (u64*)buf;
                ubuf[0] = znn_correct_endian32(*(u64*)buf);
                *(xp++) = fbuf[0];
            } break;
            default: znn_unreachable();
            }
        }
    }

    return true;
}

znn_dataset _znn_dataset_load_idx(const char *dpath, const char *lpath, const char *file, u32 line) {
    znn_trace(file, line);
    znn_dataset d = {0};
    d.data = _znn__idx_load_file(dpath);
    d.label = _znn__idx_load_file(lpath);
    assert(d.data.shape[0] == d.label.shape[0]);
    return d;
}

void _znn_dataset_destroy(znn_dataset d, const char *file, u32 line) {
    znn_trace(file, line);
    _znn__idx_destroy(d.data);
    _znn__idx_destroy(d.label);
}

bool _znn_dataset_get_batch(znn_dataset *d, u32 bs, znn_tensor *x, znn_tensor *y, const char *file, u32 line) {
    znn_trace(file, line);
    bool a = _znn__idx_get_batch(&d->data, bs, x);
    bool b = _znn__idx_get_batch(&d->label, bs, y);
    assert(a == b);
    return a;
}
