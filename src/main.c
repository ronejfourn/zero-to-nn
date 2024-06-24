#include "znn_util.h"
#include "znn_loss.h"
#include "znn_layers.h"
#include "znn_random.h"
#include "znn_dataset.h"
#include "znn_optimizers.h"

#define EPOCHS 5
#define BATCH_SIZE 64
#define LEARNING_RATE 1e-3

bool check(f32 *p, f32 *y) {
    u32 s = 0;
    for (u32 j = 0; j < BATCH_SIZE; j ++) {
        u32 mp = 0, my = 0;
        for (u32 i = 1; i < 10; i ++) {
            mp = p[j * 10 + i] > p[j * 10 + mp] ? i : mp;
            my = y[j * 10 + i] > y[j * 10 + my] ? i : my;
        }
        s += mp == my;
    }
    return s;
}

int main() {
    znn_dataset train = znn_dataset_load_idx(
            ZNN_DATASET_DIR"train-images-idx3-ubyte",
            ZNN_DATASET_DIR"train-labels-idx1-ubyte");
    znn_dataset test = znn_dataset_load_idx(
            ZNN_DATASET_DIR"t10k-images-idx3-ubyte",
            ZNN_DATASET_DIR"t10k-labels-idx1-ubyte");

#if 0

    FILE *gnuplot = popen("gnuplot", "w");
    if (!gnuplot) return 1;

    fprintf(gnuplot, "plot '-' u 1:2:3 w image \n");

    for (int i = 0; i < 28; i++)
        for (int j = 0; j < 28; j++) {
            fprintf(gnuplot, "%d %d %f\n", j + 1, i + 1, train.data.data[i * 28 + j]);
            fprintf(stdout, "%d %d %f\n", j + 1, i + 1, train.data.data[i * 28 + j]);
        }

    fprintf(gnuplot, "e\n");
    fflush(gnuplot);
    getc(stdin);
#endif

    znn_layer l[] = {
        znn_layer_flatten(),
        znn_layer_linear(28*28, 512),
        znn_layer_ReLU(),
        znn_layer_linear(512, 512),
        znn_layer_ReLU(),
        znn_layer_linear(512, 10),
    };

    znn_sequential s = znn_sequential_create(l, znn_arraylen(l));
    znn_loss mse = znn_loss_mse();
    znn_optimizer sgd = znn_optimizer_SGD(s.parameters, s.n_params, LEARNING_RATE);

    znn_tensor x, y;
    for (u32 epoch = 0; epoch < EPOCHS; epoch++) {
        printf("   Epoch %d\n", epoch + 1);
        printf("----------------------------\n");
        for (u32 batch = 0; znn_dataset_get_batch(&train, BATCH_SIZE, &x, &y); batch++) {
            znn_tensor yoh = znn_tensor_one_hot(&y, 10);
            znn_tensor_divide(&x, 255);

            znn_tensor *pred = znn_sequential_forward(&s, &x);

            znn_tensor *loss = znn_loss_calc(&mse, pred, &yoh);
            znn_tensor_backward(loss);

            znn_optimizer_step(&sgd);
            znn_optimizer_zero_grad(&sgd);

            if (batch % 100 == 0)
                printf("[%5d/%5d] loss: %f\n", 
                        batch * BATCH_SIZE + BATCH_SIZE,
                        train.data.shape[0],
                        loss->data[0]);

            znn_tensor_destroy(x);
            znn_tensor_destroy(y);
            znn_tensor_destroy(yoh);
        }

        f32 test_loss = 0, correct = 0, count = 0;
        while (znn_dataset_get_batch(&test, BATCH_SIZE, &x, &y)) {
            znn_tensor yoh = znn_tensor_one_hot(&y, 10);
            znn_tensor_divide(&x, 255);
            znn_tensor *pred = znn_sequential_forward(&s, &x);
            znn_tensor *loss = znn_loss_calc(&mse, pred, &yoh);
            test_loss += loss->data[0];
            correct += check(pred->data, yoh.data);
            count ++;

            znn_tensor_destroy(x);
            znn_tensor_destroy(y);
            znn_tensor_destroy(yoh);
        }

        printf("accuracy: %5.3f%% avg. loss: %f\n", correct / (BATCH_SIZE * count) * 100, test_loss / count);
    }

    znn_sequential_destroy(s);
    return 0;
}
