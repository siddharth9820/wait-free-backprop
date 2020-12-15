#include "cudnn_layers/generic_layer.h"
#include "cudnn_layers/convolution.h"
#include "cudnn_layers/fc.h"
#include "cudnn_layers/relu.h"
#include <iostream>
#include "common.h"
#include <unistd.h>
#include <stdint.h>
#include "dataloader.h"
#include <math.h>
#include "mpi.h"
#include "nccl.h"
 
__global__ void optimise(float * params,  float * gradient ,int N, float learning_rate)
{
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  if(i>N)return;
  params[i] -= learning_rate*gradient[i];
  gradient[i] = 0.0;
}


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

float get_loss_and_grad(int output_shape[], float * output, float * grad_output, int * labels)
{
   float loss = 0;
   for(int i=0;i<output_shape[0];i++){
	float sum = 0;
	for(int j=0;j<output_shape[1];j++){
	  sum += exp(output[i*output_shape[1]+j]);
        }
    for(int j=0;j<output_shape[1];j++){
	   float p = exp(output[i*output_shape[1]+j])/sum;
	   grad_output[i*output_shape[1]+j] = p;
           if(j == labels[i]){
		grad_output[i*output_shape[1]+j]-=1;	  
           	loss += -log(p);
           }
       }
    }
    return loss/output_shape[0];
}

int main(int argc, char* argv[])
{
    int myRank, nRanks, localRank=0;
    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));
    
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    cublasHandle_t cublas;
    cublasCreate(&cublas);


    uint64_t hostHashs[nRanks];
    char hostname[1024];
    getHostName(hostname, 1024);
    hostHashs[myRank] = getHostHash(hostname);
    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
    for (int p=0; p<nRanks; p++) {
      if (p == myRank) break;
      if (hostHashs[p] == hostHashs[myRank]) localRank++;
    }

    
    checkCUDA(cudaSetDevice(myRank));

    //Create a Simple LeNet
    Layer * network[7];
    int input_shape[4] = {64, 1, 28, 28};
    int output_shape[4];
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
    for(int i=0;i<4;i++)output_shape[i] = input_shape[i];

    //Do a forward Pass
    //Step 1 - Copy batch to GPU - Here we will generate random batch
    int input_size = network[0]->get_input_size();
    float * d_batch, *d_grad_batch ,* batch;
    checkCUDA(cudaMalloc(&d_batch, input_size));
    checkCUDA(cudaMalloc(&d_grad_batch, input_size));
    int * labels;
    //batch = (float*)malloc(input_size);
    std::normal_distribution<float> distribution(MU,SIGMA);
    std::default_random_engine generator;
    //for(int i=0; i<input_size/sizeof(float); i++)batch[i] = distribution(generator);
    MNIST_loader * loader = new MNIST_loader(64, false);
    loader->init_memory(&batch, &labels);
    loader->get_next_batch(batch, labels);
    checkCUDA(cudaMemcpy(d_batch, batch, input_size, cudaMemcpyHostToDevice));

    //Step 2 - Allocate internal memory for all layers
    for(int i=0; i<7; i++)network[i]->allocate_internal_memory();

    //Step 3 - Allocate output activation buffers for each layer
    float * output_activations[7], * grad_output_activations[7];
    for(int i=0; i<7; i++)
    {
        int output_size = network[i]->get_output_size();
        checkCUDA(cudaMalloc(&output_activations[i], output_size));
        checkCUDA(cudaMalloc(&grad_output_activations[i], output_size));
    }

    int t = 10;
    while(t--){
    //Step 4 - Do a forward Pass 
    for(int i=0;i<7;i++)
    {
        if(i==0)network[i]->forward(d_batch, output_activations[0]);
        else network[i]->forward(output_activations[i-1], output_activations[i]);
    }

    //Step 5 - Print output of final layer
    int output_size = network[6]->get_output_size();
    network[6]->get_output_shape(input_shape);
    float * output = (float*)malloc(output_size);
    checkCUDA(cudaMemcpy(output, output_activations[6], output_size, cudaMemcpyDeviceToHost));

    //std::cout << "========= Printing output of final layer ==================" << std::endl;

    //for(int i=0; i<input_shape[0]; i++){
     //   for(int j=0;j<input_shape[1];j++){
    //        std::cout << output[i*input_shape[1] + j ] << " ";
    //    }
    //    std::cout << std::endl;
    //}

    //Step 6 - Use random gradient for output right now
    float * grad_output = (float*)malloc(output_size);
    //for(int i=0; i<output_size/sizeof(float); i++) grad_output[i] = distribution(generator);
    float loss =  get_loss_and_grad(output_shape, output, grad_output, labels);
    checkCUDA(cudaMemcpy(grad_output_activations[6], grad_output, output_size, cudaMemcpyHostToDevice));
    std::cout << "==============Printing loss ============" << loss << std::endl;
    //Step 7 - Do backward Pass 
    for(int i=6; i>=0; i--)
    {
        if(i==0)network[i]->backward(grad_output_activations[i], d_grad_batch, d_batch, output_activations[i]);
        else network[i]->backward(grad_output_activations[i], grad_output_activations[i-1], output_activations[i-1], output_activations[i]);
	int param_size = network[i]->get_param_size();
	if (param_size > 0)
        	optimise<<<1, param_size/sizeof(float)>>>(network[i]->params, network[i]->params_gradients  , param_size/sizeof(float), 0.01);

    }


    //std::cout << "========= Printing gradients of layer 6 ==================" << std::endl;
    
    
    //Print gradient of Layer 6 
    //int parameter_size = network[6]->get_param_size();
    //float * gradients = (float*)malloc(parameter_size);
    //checkCUDA(cudaMemcpy(gradients, network[6]->params_gradients, parameter_size, cudaMemcpyDeviceToHost));
    //for(int i=0; i<parameter_size/sizeof(float); i++)
    //    std::cout << gradients[i] << " ";

    //std::cout << std::endl;
    }
    cudaDeviceSynchronize();
    (MPI_Finalize());
    //TODO :- now do an all reduce on gradients of all layers via NCCL
}
