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

  Network network;

  {
    const int n_hidden_layer = 1;
    const int n_neuron[] = { MNIST_IMAGE_SIZE, 64, MNIST_LABEL_SIZE };

    createNetwork(&network, 2 + n_hidden_layer, rng);

    createLayer(&network, 0, n_neuron[0]);
    for (int i = 1; i <= n_hidden_layer; i++) {
      createLayer(&network, i, n_neuron[i]);
    }
    createLayer(&network, n_hidden_layer + 1, n_neuron[n_hidden_layer + 1]);

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

        // double z[MNIST_LABEL_SIZE] = { 0., };
        // z[training_label[k]] = 1.;
        error += updateByBackPropagation(&network, training_image[k]);
      }
      printf("1: epoch: %d, error: %f\n", i, error);
      updateW(&network);
    }
  }

  Network network2;
  {
    const int n_hidden_layer = 2;
    const int n_neuron[] = { MNIST_IMAGE_SIZE, 64, 32, 64 };
    // const int n_neuron[] = { MNIST_IMAGE_SIZE, 32, 24, MNIST_LABEL_SIZE };

    createNetwork(&network2, 2 + n_hidden_layer, rng);

    createLayer(&network2, 0, n_neuron[0]);
    for (int i = 1; i <= n_hidden_layer; i++) {
      createLayer(&network2, i, n_neuron[i]);
    }
    createLayer(&network2, n_hidden_layer + 1, n_neuron[n_hidden_layer + 1]);


    for (int i = 1; i <= n_hidden_layer - 1; i++) {
      createConnection(&network2, i, sparse_random);
    }

    createConnection(&network2, 0, NULL);
    copyConnection(&network, 0, &network2, 0);

    createConnection(&network2, n_hidden_layer, NULL);
    copyConnectionWithTranspose(&network2, n_hidden_layer - 1, &network2, n_hidden_layer);

    deleteConnection(&network, 1);
    deleteConnection(&network, 0);
    deleteLayer(&network, 2);
    deleteLayer(&network, 1);
    deleteLayer(&network, 0);
    deleteNetwork(&network);

    // for (int i = 0; i < MNIST_TRAINING_DATA_SIZE; i++) {
    for (int i = 0; i < n_epoch; i++) {
      initializeDW(&network2);
      double error = 0;

      for(int j = 0; j < batch_size; j++) {
        int k = (int)(MNIST_TRAINING_DATA_SIZE * sfmt_genrand_real2(&rng));
      // {
      //   int k = i;
        setInput(&network2, training_image[k]);
        forwardPropagation(&network2, sigmoid);

        // double z[MNIST_LABEL_SIZE] = { 0., };
        // z[training_label[k]] = 1.;
        error += updateByBackPropagationPartial(&network2, network2.layer[n_hidden_layer-1].z);
      }
      printf("2: epoch: %d, error: %f\n", i, error);
      updateW(&network2);
      copyConnectionWithTranspose(&network2, n_hidden_layer - 1, &network2, n_hidden_layer);

      // scanf("%lf", &error);
    }
  }

  {
    const int n_hidden_layer = 2;
    const int n_neuron[] = { MNIST_IMAGE_SIZE, 64, 32, MNIST_LABEL_SIZE };

    deleteConnection(&network2, n_hidden_layer);
    deleteLayer(&network2, n_hidden_layer + 1);

    createLayer(&network2, n_hidden_layer + 1, n_neuron[n_hidden_layer + 1]);

    createConnection(&network2, n_hidden_layer, uniform_random);

    // for (int i = 0; i < MNIST_TRAINING_DATA_SIZE; i++) {
    for (int i = 0; i < n_epoch; i++) {
      initializeDW(&network2);
      double error = 0;

      for(int j = 0; j < batch_size; j++) {
        int k = (int)(MNIST_TRAINING_DATA_SIZE * sfmt_genrand_real2(&rng));
      // {
      //   int k = i;
        setInput(&network2, training_image[k]);
        forwardPropagation(&network2, sigmoid);

        double z[MNIST_LABEL_SIZE] = { 0., };
        z[training_label[k]] = 1.;
        error += updateByBackPropagation(&network2, z);
      }

      printf("3: epoch: %d, error: %f\n", i, error);
      updateW(&network2);
    }
  }


  { // Evaluation
    Layer *output_layer = &network2.layer[network2.n - 1];
    const int n = output_layer->n;
    int correct = 0;
    for (int k = 0; k < MNIST_TEST_DATA_SIZE; k++) {
      setInput(&network2, test_image[k]);
      forwardPropagation(&network2, sigmoid);
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

  deleteConnection(&network2, 2);
  deleteConnection(&network2, 1);
  deleteConnection(&network2, 0);
  deleteLayer(&network2, 3);
  deleteLayer(&network2, 2);
  deleteLayer(&network2, 1);
  deleteLayer(&network2, 0);
  deleteNetwork(&network2);

  return 0;
}
