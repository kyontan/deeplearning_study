# LIBGD=/home3/staff/ya000836/.ced/usr

all: skel perceptron bp_xor bp_mnist ae_mnist mnist_test

skel: skel.c SFMT-src-1.5.1/SFMT.c
	clang-omp -O3 -std=c99 -Wall -I SFMT-src-1.5.1 -D SFMT_MEXP=19937 -o $@ $^ -lm

perceptron: perceptron.c SFMT-src-1.5.1/SFMT.c
	clang-omp -O3 -std=c99 -Wall -I SFMT-src-1.5.1 -D SFMT_MEXP=19937 -o $@ $^ -lm

bp.o: bp.c SFMT-src-1.5.1/SFMT.c
	clang-omp -O3 -std=c99 -Wall -I SFMT-src-1.5.1 -D SFMT_MEXP=19937 -c $^

bp_xor: bp_xor.c bp.c SFMT-src-1.5.1/SFMT.c
	clang-omp -O3 -std=c99 -Wall -I SFMT-src-1.5.1 -D SFMT_MEXP=19937 -o $@ $^ -lgd

bp_mnist: bp_mnist.c SFMT-src-1.5.1/SFMT.c mnist.c bp.o
	clang-omp -O3 -std=c99 -Wall -I SFMT-src-1.5.1 -D SFMT_MEXP=19937 -o $@ $^ -lgd

ae_mnist: ae_mnist.c SFMT-src-1.5.1/SFMT.c mnist.c bp.o
	clang-omp -O3 -g -std=c99 -Wall -I SFMT-src-1.5.1 -D SFMT_MEXP=19937 -o $@ $^ -lgd

mnist_test: mnist_test.c mnist.c
	clang-omp -O3 -std=c99 -Wall -o $@ $^ -lgd

clean:
	rm -f *.o skel perceptron bp bp_xor mnist_test bp_mnist ae_mnist
