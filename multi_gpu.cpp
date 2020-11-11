#include "cudnn_layers/generic_layer.h"
#include "cudnn_layers/convolution.h"
#include "cudnn_layers/fc.h"
#include "cudnn_layers/relu.h"
#include "common.h"

#include "mpi.h"
#include "cuda_runtime.h"

#include <unistd.h>
#include <iostream>

static uint64_t getHostHash(const char* string) {
    // Based on DJB2, result = result * 33 + char
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++){
        result = ((result << 5) + result) + string[c];
    }
    return result;
}

static void getHostName(char* hostname, int maxlen) {
    gethostname(hostname, maxlen);
    for (int i=0; i< maxlen; i++) {
        if (hostname[i] == '.') {
            hostname[i] = '\0';
            return;
        }
    }
}

//calculating local_rank which is used in selecting a GPU
static int get_local_rank(int my_rank, int n_ranks) {
    int local_rank = 0;
    uint64_t hostHashs[n_ranks];
    char hostname[1024];
    getHostName(hostname, 1024);
    hostHashs[my_rank] = getHostHash(hostname);
    MPICHECK(MPI_Allgather(
        MPI_IN_PLACE,
        0,
        MPI_DATATYPE_NULL,
        hostHashs,
        sizeof(uint64_t),
        MPI_BYTE,
        MPI_COMM_WORLD
    ));

    for (int p=0; p<n_ranks; p++) {
        if (p == my_rank) break;
        if (hostHashs[p] == hostHashs[my_rank]) local_rank++;
    }

    return local_rank;
}

int main(int argc, char* argv[])
{
    int size = 32*1024*1024;

    int my_rank, n_ranks, local_rank = 0;

    //initializing MPI
    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &my_rank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &n_ranks));
 
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    cublasHandle_t cublas;
    cublasCreate(&cublas);

    // Assume each rank gets one gpu for now
    local_rank = get_local_rank(my_rank, n_ranks);
    CUDACHECK(cudaSetDevice(local_rank));
    
    // int filter_size[3] = {3, 3, 3};        //HWC
    // int input_shape[4] = {64, 1, 10, 10};  //NCHW
    // check_conv(cudnn, filter_size, input_shape);
    // check_FC(cublas, 100, input_shape);
    // check_ReLU(cudnn, input_shape);

    //Create a Simple LeNet
    Layer * network[7];
    int input_shape[4] = {64, 1, 28, 28};
    int filter1[3] = {5, 5, 32};
    int filter2[3] = {5, 5, 64};
    
    network[0] = new Convolution(filter1, input_shape, cudnn);
    network[0]->get_output_shape(input_shape);
    
    network[1] = new ReLU(input_shape, cudnn);
    network[1]->get_output_shape(input_shape);

    network[2] = new Convolution(filter2, input_shape, cudnn);
    network[2]->get_output_shape(input_shape);

    network[3] = new ReLU(input_shape, cudnn);
    network[3]->get_output_shape(input_shape);

    network[4] = new FC(50, input_shape, cublas);
    network[4]->get_output_shape(input_shape);

    network[5] = new ReLU(input_shape, cudnn);
    network[5]->get_output_shape(input_shape);

    network[6] = new FC(10, input_shape, cublas);
    network[6]->get_output_shape(input_shape);

    std::cout <<"Shape of output : " << input_shape[0] << " " << input_shape[1] << " " << input_shape[2] << " " << input_shape[3] << std::endl;

    //Do a forward Pass
    //Step 1 - Copy batch to GPU - Here we will generate random batch
    int input_size = network[0]->get_input_size();
    float *d_batch, *d_grad_batch, *batch;
    checkCUDA(cudaMalloc(&d_batch, input_size));
    checkCUDA(cudaMalloc(&d_grad_batch, input_size));
    batch = (float*)malloc(input_size);
    std::normal_distribution<float> distribution(MU,SIGMA);
    std::default_random_engine generator;
    for(int i=0; i<input_size/sizeof(float); i++)batch[i] = distribution(generator);
    checkCUDA(cudaMemcpy(d_batch, batch, input_size, cudaMemcpyHostToDevice));

    //Step 2 - Allocate internal memory for all layers
    for(int i=0; i<7; i++)network[i]->allocate_internal_memory();

    //Step 3 - Allocate output activation buffers for each layer
    float *output_activations[7], *grad_output_activations[7];
    for(int i=0; i<7; i++)
    {
        int output_size = network[i]->get_output_size();
        checkCUDA(cudaMalloc(&output_activations[i], output_size));
        checkCUDA(cudaMalloc(&grad_output_activations[i], output_size));
    }

    //Step 4 - Do a forward Pass 
    
    // first layer is special
    network[0]->forward(d_batch, output_activations[0]);
    for(int i=1;i<7;i++)
    {
        network[i]->forward(output_activations[i-1], output_activations[i]);
    }

    //Step 5 - Print output of final layer
    int output_size = network[6]->get_output_size();
    network[6]->get_output_shape(input_shape);
    float * output = (float*)malloc(output_size);
    checkCUDA(cudaMemcpy(output, output_activations[6], output_size, cudaMemcpyDeviceToHost));

    std::cout << "========= Printing output of final layer ==================" << std::endl;

    for(int i=0; i<input_shape[0]; i++){
        for(int j=0;j<input_shape[1];j++){
            std::cout << output[i*input_shape[1] + j ] << " ";
        }
        std::cout << std::endl;
    }

    //Step 6 - Use random gradient for output right now
    float * grad_output = output;
    for(int i=0; i<output_size/sizeof(float); i++)
        grad_output[i] = distribution(generator);
    checkCUDA(cudaMemcpy(grad_output_activations[6], grad_output, output_size, cudaMemcpyHostToDevice));

    //Step 7 - Do backward Pass 
    for(int i=6; i>0; i--)
    {
        network[i]->backward(
            grad_output_activations[i],
            grad_output_activations[i-1],
            output_activations[i-1],
            output_activations[i]
        );
    }
    // first layer is special
    network[0]->backward(
        grad_output_activations[0],
        d_grad_batch,
        d_batch,
        output_activations[0]
    );

    std::cout << "========= Printing gradients of layer 6 ==================" << std::endl;
    
    //Print gradient of Layer 6 
    int parameter_size = network[6]->get_param_size();
    float * gradients = (float*)malloc(parameter_size);
    checkCUDA(cudaMemcpy(gradients, network[6]->params_gradients, parameter_size, cudaMemcpyDeviceToHost));
    for(int i=0; i<parameter_size/sizeof(float); i++)
        std::cout << gradients[i] << " ";

    std::cout << std::endl;

    //TODO :- now do an all reduce on gradients of all layers via NCCL
    
}
