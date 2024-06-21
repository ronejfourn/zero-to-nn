#include "zm_types.h"
#include "zm_tensor.h"

#include <stdio.h>

int main() {
    float a[2][3] = {
        {1, 2, 3},
        {3, 2, 1},
    };

    u32 s[] = {2,3};
    zm_tensor t = zm_tensor_create(sizeof(s) / sizeof(s[0]), s, a);
    zm_tensor_print(&t);
}
