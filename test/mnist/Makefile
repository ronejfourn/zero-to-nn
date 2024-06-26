TARGET_DIR := bin
TARGET := mnist

SRC_DIRS := . ../..
DATASET_DIR := $(shell realpath ./res/)
DATASETS := train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte

DATASET_PATH := $(DATASETS:%=$(DATASET_DIR)/%)

CFLAGS := -DDATASET_DIR=$(DATASET_DIR) -DZNN_OPENMP_ENABLE=1
CFLAGS += -march=native -O3 -fopenmp -ffast-math
CFLAGS += -Wall -pedantic

LDFLAGS := -lm -fopenmp -ffast-math

SRCS := $(shell find $(SRC_DIRS) -maxdepth 1 -name '*.c')

$(TARGET_DIR)/$(TARGET): $(SRCS) $(DATASET_PATH)
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(SRCS) -o $@ $(LDFLAGS)

$(DATASET_PATH):
	mkdir -p $(dir $@)
	curl -L https://storage.googleapis.com/cvdf-datasets/mnist/$(notdir $@).gz -o $@.gz
	gzip -d $@.gz

.PHONY: clean
clean:
	rm -rf $(TARGET_DIR)
