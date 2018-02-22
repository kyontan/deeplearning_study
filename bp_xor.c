#include "bp.h"
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char **argv) {
  sfmt_t rng;
  sfmt_init_gen_rand(&rng, getpid());

  if (argc != 2) {
    fprintf(stderr, "%s middle_layer_neurons\n", argv[0]);
    return 1;
  }
  const int middle_layer_neurons = atoi(argv[1]);

  Network network;
  createNetwork(&network, 3, rng);
  createLayer(&network, 0, 2);
  // createLayer(&network, 1, 2);
  createLayer(&network, 1, middle_layer_neurons);
  createLayer(&network, 2, 1);
  createConnection(&network, 0, sparse_random);
  createConnection(&network, 1, uniform_random);

  Neuron x[4][ 2] = { { 0., 0. }, { 0., 1. }, { 1., 0. }, { 1., 1. } };
  Neuron z[4][ 1] = { { 0. } , { 1. } , { 1. } , { 0.} };
  const int number_of_training_data = 4;

  // Training
  float error = 1.0; // arbitrary large number
  const float Epsilon = 0.001; // tolerance
  int i = 0;

  dump_network(&network);
  return 0;

  while(error > Epsilon) {
    error = 0.;
    initializeDW(&network);
    for(int j = 0; j < number_of_training_data; j++) {
      //int k = (int)(number_of_training_data * sfmt_genrand_real2(&rng));
      int k = j;
      setInput(&network, x[k]);
      forwardPropagation(&network, sigmoid);
      error += updateByBackPropagation(&network, z[k]);
    }
    updateW(&network);
    printf("%d %f\n", i, error);
    i++;
  }
  fprintf(stderr, "# of epochs = %d\n", i);

  // Test
  Layer *output_layer = &network.layer[network. n - 1];
  const int n = output_layer->n;
  for(int i = 0; i < number_of_training_data; i++) {
    setInput(&network, x[i]);
    forwardPropagation(&network, sigmoid);
    //dump(&network);
    for(int j = 0; j < n; j++) {
      fprintf(stderr, "%f%s", output_layer->z[j],(j == n - 1) ? "\n" : " ");
    }
  }

  dump_network(&network);

  deleteConnection(&network, 1);
  deleteConnection(&network, 0);
  deleteLayer(&network, 2);
  deleteLayer(&network, 1);
  deleteLayer(&network, 0);
  deleteNetwork(&network);

  return 0;
}
