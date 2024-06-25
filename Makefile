TARGET_EXEC := znn_mnist

OBJ_DIR := ./obj
SRC_DIRS := ./src
DATASET_DIR := $(shell realpath ./res/)
DATASETS := train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte

DATASET_PATH := $(DATASETS:%=$(DATASET_DIR)/%)

CFLAGS := -DZNN_DATASET_DIR=\"$(DATASET_DIR)/\" -DZNN_OPENMP_ENABLE=1
CFLAGS += -march=native -O3 -fopenmp -ffast-math
CFLAGS += -Wall -pedantic

LDFLAGS := -lm -fopenmp -ffast-math

SRCS := $(shell find $(SRC_DIRS) -name '*.c')
OBJS := $(SRCS:%=$(OBJ_DIR)/%.o)

$(TARGET_EXEC): $(OBJS)
	$(CC) $(OBJS) -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.c.o: %.c $(DATASET_PATH)
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(DATASET_PATH):
	mkdir -p $(dir $@)
	curl -L https://storage.googleapis.com/cvdf-datasets/mnist/$(notdir $@).gz -o $@.gz
	gzip -d $@.gz

.PHONY: clean
clean:
	rm -rf $(OBJ_DIR)
	rm $(TARGET_EXEC)
