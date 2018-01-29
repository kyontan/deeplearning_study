#include <stdio.h>
#include <stdlib.h>
// #include <math.h>
#include <unistd.h>
#include "mnist.h"
#include "bp.h"

int main(int argc, char **argv) {
  // const int n_epoch = 10000;
  // const int batch_size = 100;
  if (argc != 3) {
    printf("%s batch_size epoch_count\n", argv[0]);
    return 1;
  }

  int batch_size = atoi(argv[1]);
  int n_epoch = atoi(argv[2]);

  double **training_image, **test_image;
  int *training_label, *test_label;
  mnist_initialize(&training_image, &training_label, &test_image, &test_label);

  sfmt_t rng;
  sfmt_init_gen_rand(&rng, getpid());

  const int n_hidden_layer = 1;
  const int n_hidden_layer_neuron = 64;

  Network network;
  createNetwork(&network, 2 + n_hidden_layer, rng);

  createLayer(&network, 0, MNIST_IMAGE_SIZE);
  for (int i = 1; i <= n_hidden_layer; i++) {
    createLayer(&network, i, n_hidden_layer_neuron);
  }
  createLayer(&network, n_hidden_layer + 1, MNIST_LABEL_SIZE);

  for (int i = 0; i < n_hidden_layer; i++) {
    createConnection(&network, i, sparse_random);
  }
  createConnection(&network, n_hidden_layer, uniform_random);

  // for (int i = 0; i < MNIST_TRAINING_DATA_SIZE; i++) {
  for (int i = 0; i < n_epoch; i++) {
    initializeDW(&network);
    double error = 0;

    for(int j = 0; j < batch_size; j++) {
      int k = (int)(MNIST_TRAINING_DATA_SIZE * sfmt_genrand_real2(&rng));
    // {
    //   int k = i;
      setInput(&network, training_image[k]);
      forwardPropagation(&network, sigmoid);

      double z[MNIST_LABEL_SIZE] = { 0., };
      z[training_label[k]] = 1.;
      error += updateByBackPropagation(&network, z);
    }
    printf("epoch: %d, error: %f\n", i, error);
    updateW(&network);
  }

  { // Evaluation
    Layer *output_layer = &network.layer[network.n - 1];
    const int n = output_layer->n;
    int correct = 0;
    for (int k = 0; k < MNIST_TEST_DATA_SIZE; k++) {
      setInput(&network, test_image[k]);
      forwardPropagation(&network, sigmoid);
      int maxj = 0;
      double maxz = 0.;

      for (int j = 0; j < n; j++){
        if (maxz < output_layer->z[j]) {
          maxz = output_layer->z[j];
          maxj = j;
        }
      }

      correct += (maxj == test_label[k]);
    }

    fprintf(stderr, "success_rate = %f\n", (double)correct / MNIST_TEST_DATA_SIZE);
  }

  mnist_finalize(training_image, training_label, test_image, test_label);

  return 0;
}
