#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include "bp.h"

extern void sfmt_init_gen_rand(sfmt_t * sfmt, uint32_t seed);
extern double sfmt_genrand_real2(sfmt_t * sfmt);

double all_to_all(Network *n, const int i, const int j) { return 1.; }
double uniform_random(Network *n, const int i, const int j) { return 1. - 2. * sfmt_genrand_real2(&n->rng); }
double sparse_random(Network *n, const int i, const int j) {
  return(sfmt_genrand_real2(&n->rng) < 0.5) ? uniform_random(n, i, j) : 0.;
}
double sigmoid(double x) { return 1. /(1. + exp(- x)); }
double relu(double x) { return (0 < x) ? x : 0; }

void createNetwork(Network *network, const int number_of_layers, const sfmt_t rng) {
  network->layer = (Layer *) malloc(number_of_layers * sizeof(Layer));
  network->connection = (Connection *) malloc(number_of_layers * sizeof(Connection));
  network->n = number_of_layers;
  network->rng = rng;
}

void deleteNetwork(Network *network) {
  free(network->layer);
  free(network->connection);
}

void createLayer(Network *network, const int layer_id, const int number_of_neurons) {
  Layer *layer = &network->layer[layer_id];

  layer->n = number_of_neurons;

  int bias = (layer_id < network->n - 1) ? 1 : 0; // 出力層以外はバイアスを用意

  layer->z = (Neuron *) malloc((number_of_neurons + bias) * sizeof(Neuron));
  for(int i = 0; i < layer->n; i++) { layer->z[i] = 0.; } // 初期化
  if(bias) { layer->z[layer->n] = +1.; } // バイアス初期化

  // Deltaを追加
  layer->delta = (Delta *) malloc((number_of_neurons + bias) * sizeof(Delta));
  for(int i = 0; i < layer->n; i++) { layer->delta[i] = 0.; }
    if(bias) { layer->delta[layer->n] = 0.; } // バイアス初期化
}

void deleteLayer(Network *network, const int layer_id) {
  Layer *layer = &network->layer[layer_id];
  free(layer->z);
  free(layer->delta);
}

void createConnection(Network *network, const int layer_id, double(*func)(Network *, const int, const int)) {
  Connection *connection = &network->connection[layer_id];

  const int n_pre = network->layer[layer_id].n + 1; // +1 for bias
  const int n_post = (layer_id == network->n - 1) ? 1 : network->layer[layer_id + 1].n;

  connection->w = (Weight *) malloc(n_pre * n_post * sizeof(Weight));
  connection->dw = (Weight *) malloc(n_pre * n_post * sizeof(Weight));

  if (func != NULL) {
    for(int i = 0; i < n_post; i++) {
      for(int j = 0; j < n_pre; j++) {
        connection->w[n_pre * i + j] = func(network, i, j);
        connection->dw[n_pre * i + j] = 0.;
      }
    }
  }

  connection->n_pre = n_pre;
  connection->n_post = n_post;
}

void deleteConnection(Network *network, const int layer_id) {
  Connection *connection = &network->connection[layer_id];
  free(connection->w);
  free(connection->dw);
}

void copyConnection(const Network *src_network, const int src_layer_id, Network *dst_network, const int dst_layer_id) {
  Connection *src_connection = &src_network->connection[src_layer_id];
  Connection *dst_connection = &dst_network->connection[dst_layer_id];

  const int n_pre = src_network->layer[src_layer_id].n + 1; // +1 for bias
  const int n_post = (src_layer_id == src_network->n - 1) ? 1 : src_network->layer[src_layer_id + 1].n;

  for(int i = 0; i < n_post; i++) {
    for(int j = 0; j < n_pre; j++) {
      dst_connection->w[n_pre * i + j] = src_connection->w[n_pre * i + j];
    }
  }
}

void copyConnectionWithTranspose(const Network *src_network, const int src_layer_id, Network *dst_network, const int dst_layer_id) {
  Connection *src_connection = &src_network->connection[src_layer_id];
  Connection *dst_connection = &dst_network->connection[dst_layer_id];

  const int src_n_pre = src_network->layer[src_layer_id].n + 1; // +1 for bias
  const int dst_n_pre = dst_network->layer[dst_layer_id].n + 1; // +1 for bias
  const int n_post = (src_layer_id == src_network->n - 1) ? 1 : src_network->layer[src_layer_id + 1].n;

  for(int i = 0; i < n_post; i++) {
    for(int j = 0; j < src_n_pre - 1; j++) { // skipping bias
      dst_connection->w[dst_n_pre * j + i] = src_connection->w[src_n_pre * i + j];
    }
  }
}

void setInput(Network *network, Neuron x[]) {
  Layer *input_layer = &network->layer[0];
  for(int i = 0; i < input_layer->n; i++) {
    input_layer->z[i] = x[i];
  }
}

void forwardPropagation(Network *network, double(*activation)(double)) {
  for(int i = 0; i < network->n - 1; i++) {
    Layer *l_pre = &network->layer[i];
    Layer *l_post = &network->layer[i + 1];
    Connection *c = &network->connection[i];
    for(int j = 0; j < c->n_post; j++) {
      Neuron u = 0.;
      for(int k = 0; k < c->n_pre; k++) {
        u += (c->w[k + c->n_pre * j]) * (l_pre->z[k]);
      }
      l_post->z[j] = activation(u);
    }
  }
}

// update output layers
// @returns error
double updateByBackPropagationOutputLayer(Network *network, Neuron z[], const double Eta) {
  double error = 0;

  // output layer's delta
  Layer *layer = &network->layer[network->n - 1];

  for(int i = 0; i < layer->n; i++) { // output layer's neuron size == length of z[]
    layer->delta[i] = z[i] - layer->z[i];
    error += 0.5 * pow(z[i] - layer->z[i], 2);
  }

  return error;
}

// update hidden (not output) layers
void updateByBackPropagationInLayer(Network *network, Neuron z[], const int layer_idx, const double Eta) {
  Layer *l_pre = &network->layer[layer_idx];
  Layer *l_post = &network->layer[layer_idx + 1];
  Connection *c = &network->connection[layer_idx];

  const int n_pre = l_pre->n + 1; // +1 for bias
  const int n_post = l_post->n;

  // update dw
  for(int j = 0; j < n_post; j++) {
    for(int i = 0; i < n_pre; i++) {
      c->dw[n_pre * j + i] += Eta * l_post->delta[j] * (l_post->z[j] * (1 - l_post->z[j]) * l_pre->z[i]);
    }
  }

  // update hidden layer's delta
  for(int i = 0; i < n_pre; i++) {
    l_pre->delta[i] = 0.;
    for(int k = 0; k < n_post; k++) {
      l_pre->delta[i] += l_post->delta[k] * l_post->z[k] * (1 - l_post->z[k]) * c->w[n_pre * k + i];
    }
  }
}

double updateByBackPropagation(Network *network, Neuron z[]) {
  const double Eta = 0.15;

  // output layer's delta
  double error = updateByBackPropagationOutputLayer(network, z, Eta);

  // hidden layer
  for(int layer_idx = network->n - 2; 0 <= layer_idx; layer_idx--) {
    updateByBackPropagationInLayer(network, z, layer_idx, Eta);
  }

  return error;
}

double updateByBackPropagationPartial(Network *network, Neuron z[]) {
  const double Eta = 0.15;

  // output layer's delta
  double error = updateByBackPropagationOutputLayer(network, z, Eta);
  updateByBackPropagationInLayer(network, z, network->n - 2, Eta);

  return error;
}

void initializeDW(Network *network) {
  for(int layer_id = 0; layer_id < network->n - 1; layer_id++) {
    Connection *c = &network->connection[layer_id];
    for(int i = 0; i < c->n_post; i++) {
      for(int j = 0; j < c->n_pre; j++) {
       c->dw[c->n_pre * i + j] = 0.;
     }
   }
 }
}

void updateW(Network *network) {
  for(int layer_id = 0; layer_id < network->n - 1; layer_id++) {
    Connection *c = &network->connection[layer_id];
    for(int i = 0; i < c->n_post; i++) {
      for(int j = 0; j < c->n_pre; j++) {
       c->w[c->n_pre * i + j] += c->dw[c->n_pre * i + j];
     }
   }
 }
}

void dump_neuron(int i, Neuron *neuron) {
  printf("    - neuron[%d]: %lf\n", i, *neuron);
}

void dump_layer(Layer *layer) {
  for(int i = 0; i < layer->n; i++) { dump_neuron(i, &layer->z[i]); }
}

void dump_weight(Weight *weight) {
  printf("    weight: %lf\n", *weight);
}

void dump_connection(Connection *connection) {
  for(int j = 0; j < connection->n_pre; j++) {
    for(int i = 0; i < connection->n_post; i++) {
      printf("    %d -> %d: %lf\n", j, i, connection->w[j + connection->n_pre * i]);
    }
  }
}

void dump_network(Network *network) {
  printf("network:\n");
  printf("  n: %d\n", network->n);
  for(int i = 0; i < network->n; i++) {
    printf("  layer[%d]:\n", i);
    dump_layer(&network->layer[i]);
    printf("  connection[%d]:\n", i);
    dump_connection(&network->connection[i]);
  }

}

void dump(Network *network) {
  dump_network(network);
}

int bp_local_main(void) {
  sfmt_t rng;
  sfmt_init_gen_rand(&rng, getpid());

  Network network;
  createNetwork(&network, 3, rng);
  createLayer(&network, 0, 2);
  // createLayer(&network, 1, 2);
  createLayer(&network, 1, 128);
  createLayer(&network, 2, 1);
  createConnection(&network, 0, sparse_random);
  createConnection(&network, 1, uniform_random);

  Neuron x[4][ 2] = { { 0., 0. }, { 0., 1. }, { 1., 0. }, { 1., 1. } };
  Neuron z[4][ 1] = { { 0. } , { 1. } , { 1. } , { 0.} };
  const int number_of_training_data = 4;

  // Training
  double error = 1.0; // arbitrary large number
  const double Epsilon = 0.001; // tolerance
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
  Layer *output_layer = &network . layer[network. n - 1];
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

#if 0
int main(void)
{
  return bp_local_main();
}
#endif

