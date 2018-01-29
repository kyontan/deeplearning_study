#ifndef _BP_H_
#define _BP_H_

#include <SFMT.h>

typedef float Neuron, Delta, Weight;
typedef struct { Weight *w; Weight *dw; int n_pre; int n_post; } Connection;
typedef struct { Neuron *z; Delta *delta; int n; } Layer;
typedef struct { Layer *layer; Connection *connection; sfmt_t rng; int n; } Network;

float all_to_all(Network *n, const int i, const int j);
float uniform_random(Network *n, const int i, const int j);
float sparse_random(Network *n, const int i, const int j);
float sigmoid(float x);
float relu(float x);


void createNetwork(Network *network, const int number_of_layers, const sfmt_t rng);
void deleteNetwork(Network *network);

void createLayer(Network *network, const int layer_id, const int number_of_neurons);
void deleteLayer(Network *network, const int layer_id);

void createConnection(Network *network, const int layer_id, float(*func)(Network *, const int, const int));
void deleteConnection(Network *network, const int layer_id);
void copyConnection(const Network *src_network, const int src_layer_id, Network *dst_network, const int dst_layer_id);
void copyConnectionWithTranspose(const Network *src_network, const int src_layer_id, Network *dst_network, const int dst_layer_id);

void setInput(Network *network, Neuron x[]);

void forwardPropagation(Network *network, float(*activation)(float));
float updateByBackPropagation(Network *network, Neuron z[]);
float updateByBackPropagationPartial(Network *network, Neuron z[]);

void initializeDW(Network *network);
void updateW(Network *network);


void dump_network(Network *network);

#endif // _BP_H_
