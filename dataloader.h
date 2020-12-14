#include "mnist_loader.h"
#include <iostream>
#include <stdlib.h>

class MNIST_loader
{
    public:
        int batch_size;
        int rows;
        int cols;
        int train_size;
        int test_size;
        int curr;
        mnist_loader *train;
        mnist_loader *test; 
        
    MNIST_loader(int batch_size, bool show_samples=false);
    void get_input_shape(int shape[]);
    void init_memory(float ** batch, int **labels);
    void get_next_batch(float * batch, int * labels);
    void visualize(float * batch, int * labels, int num = 1);
    void visualize(std::vector<double> image);
};