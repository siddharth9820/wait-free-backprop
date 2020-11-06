#include "cudnn_layers/generic_layer.h"
#include "cudnn_layers/convolution.h"
#include "cudnn_layers/fc.h"
#include "cudnn_layers/relu.h"
#include <iostream>
#include "common.h"

void check_conv(cudnnHandle_t cudnn, int filter_size[], int input_shape[])
{
    Layer * conv_layer = new Convolution(filter_size, input_shape, cudnn);
    
    std::cout << conv_layer->get_type() << std::endl;
    int output_shape[4];
    conv_layer->get_output_shape(output_shape);

    std::cout << "Output Shape" << std::endl;
    for(int i=0;i<4;i++)std::cout << output_shape[i] << " " ;
    std::cout << std::endl;

    std::cout << "Workspace Size: " << conv_layer->get_workspace_size() << std::endl;
    std::cout << "Input Size: " << conv_layer->get_input_size() << std::endl;
    std::cout << "Output Size: " << conv_layer->get_output_size() << std::endl;
    conv_layer->allocate_internal_memory();

    std::normal_distribution<float> distribution(MU,SIGMA);
    std::default_random_engine generator;

    float * input, * output, * grad_output, * grad_input, * d_input, * d_output, * d_grad_input, * d_grad_output;
    int input_size = conv_layer->get_input_size(), output_size = conv_layer->get_output_size();
    
    input = (float*)malloc(input_size);
    grad_output = (float*)malloc(output_size);

    for(int i=0;i<input_size/sizeof(float);i++){
        input[i] = distribution(generator);
    }

    for(int i=0;i<output_size/sizeof(float);i++){
        grad_output[i] = distribution(generator);
    }
    output = (float*)malloc(output_size);
    grad_input = (float*)malloc(input_size);

    checkCUDA(cudaMalloc(&d_input, input_size));
    checkCUDA(cudaMalloc(&d_output, output_size));
    checkCUDA(cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice));
    
    checkCUDA(cudaMalloc(&d_grad_input, input_size));
    checkCUDA(cudaMalloc(&d_grad_output, output_size));
    checkCUDA(cudaMemcpy(d_grad_output, grad_output, output_size, cudaMemcpyHostToDevice));
    
    conv_layer->forward(d_input, d_output);
    conv_layer->backward(d_grad_output, d_grad_input, d_input, d_output);

    checkCUDA(cudaMemcpy(grad_input, d_grad_input, input_size, cudaMemcpyDeviceToHost));
}

void check_FC(cublasHandle_t cublas, int output_dim, int input_shape[])
{
    Layer * fc_layer = new FC(output_dim, input_shape, cublas);
    
    std::cout << fc_layer->get_type() << std::endl;
    int output_shape[4];
    fc_layer->get_output_shape(output_shape);

    std::cout << "Output Shape" << std::endl;
    for(int i=0;i<4;i++)std::cout << output_shape[i] << " " ;
    std::cout << std::endl;

    std::cout << "Workspace Size: " << fc_layer->get_workspace_size() << std::endl;
    std::cout << "Input Size: " << fc_layer->get_input_size() << std::endl;
    std::cout << "Output Size: " << fc_layer->get_output_size() << std::endl;
    fc_layer->allocate_internal_memory();

    std::normal_distribution<float> distribution(MU,SIGMA);
    std::default_random_engine generator;

    float * input, * output, * grad_output, * grad_input, * d_input, * d_output, * d_grad_input, * d_grad_output;
    int input_size = fc_layer->get_input_size(), output_size = fc_layer->get_output_size();
    
    input = (float*)malloc(input_size);
    grad_output = (float*)malloc(output_size);

    for(int i=0;i<input_size/sizeof(float);i++){
        input[i] = distribution(generator);
    }

    for(int i=0;i<output_size/sizeof(float);i++){
        grad_output[i] = distribution(generator);
    }
    output = (float*)malloc(output_size);
    grad_input = (float*)malloc(input_size);

    checkCUDA(cudaMalloc(&d_input, input_size));
    checkCUDA(cudaMalloc(&d_output, output_size));
    checkCUDA(cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice));
    
    checkCUDA(cudaMalloc(&d_grad_input, input_size));
    checkCUDA(cudaMalloc(&d_grad_output, output_size));
    checkCUDA(cudaMemcpy(d_grad_output, grad_output, output_size, cudaMemcpyHostToDevice));
    
    fc_layer->forward(d_input, d_output);
    fc_layer->backward(d_grad_output, d_grad_input, d_input, d_output);

    // checkCUDA(cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost));
    
    // for(int i=0;i<output_size/sizeof(float);i++) std::cout << output[i] << " ";
    // std::cout << std::endl;

    checkCUDA(cudaMemcpy(grad_input, d_grad_input, input_size, cudaMemcpyDeviceToHost));
    
    // for(int i=0;i<input_size/sizeof(float);i++) std::cout << grad_input[i] << " ";
    // std::cout << std::endl;

}

void check_ReLU(cudnnHandle_t cudnn, int input_shape[])
{
    Layer * relu_layer = new ReLU(input_shape, cudnn);
    
    std::cout << relu_layer->get_type() << std::endl;
    int output_shape[4];
    relu_layer->get_output_shape(output_shape);

    std::cout << "Output Shape" << std::endl;
    for(int i=0;i<4;i++)std::cout << output_shape[i] << " " ;
    std::cout << std::endl;

    std::cout << "Workspace Size: " << relu_layer->get_workspace_size() << std::endl;
    std::cout << "Input Size: " << relu_layer->get_input_size() << std::endl;
    std::cout << "Output Size: " << relu_layer->get_output_size() << std::endl;
    relu_layer->allocate_internal_memory();

    std::normal_distribution<float> distribution(MU,SIGMA);
    std::default_random_engine generator;

    float * input, * output, * grad_output, * grad_input, * d_input, * d_output, * d_grad_input, * d_grad_output;
    int input_size = relu_layer->get_input_size(), output_size = relu_layer->get_output_size();
    
    input = (float*)malloc(input_size);
    grad_output = (float*)malloc(output_size);

    for(int i=0;i<input_size/sizeof(float);i++){
        input[i] = distribution(generator);
    }

    for(int i=0;i<output_size/sizeof(float);i++){
        grad_output[i] = distribution(generator);
    }
    output = (float*)malloc(output_size);
    grad_input = (float*)malloc(input_size);

    checkCUDA(cudaMalloc(&d_input, input_size));
    checkCUDA(cudaMalloc(&d_output, output_size));
    checkCUDA(cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice));
    
    checkCUDA(cudaMalloc(&d_grad_input, input_size));
    checkCUDA(cudaMalloc(&d_grad_output, output_size));
    checkCUDA(cudaMemcpy(d_grad_output, grad_output, output_size, cudaMemcpyHostToDevice));
    
    relu_layer->forward(d_input, d_output);
    relu_layer->backward(d_grad_output, d_grad_input, d_input, d_output);

    // checkCUDA(cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost));
    
    // for(int i=0;i<output_size/sizeof(float);i++) std::cout << output[i] << " ";
    // std::cout << std::endl;

    // checkCUDA(cudaMemcpy(grad_input, d_grad_input, input_size, cudaMemcpyDeviceToHost));
    
    // for(int i=0;i<input_size/sizeof(float);i++) std::cout << grad_input[i] << " ";
    // std::cout << std::endl;
}



int main(int argc, const char* argv[])
{
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    cublasHandle_t cublas;
    cublasCreate(&cublas);

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
    float * d_batch, * batch;
    checkCUDA(cudaMalloc(&d_batch, input_size));
    batch = (float*)malloc(input_size);
    std::normal_distribution<float> distribution(MU,SIGMA);
    std::default_random_engine generator;
    for(int i=0; i<input_size/sizeof(float); i++)batch[i] = distribution(generator);
    checkCUDA(cudaMemcpy(d_batch, batch, input_size, cudaMemcpyHostToDevice));

    //Step 2 - Allocate internal memory for all layers
    for(int i=0; i<7; i++)network[i]->allocate_internal_memory();

    //Step 3 - Allocate output activation buffers for each layer
    float * output_activations[7];
    for(int i=0; i<7; i++)
    {
        int output_size = network[i]->get_output_size();
        checkCUDA(cudaMalloc(&output_activations[i], output_size));
    }

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

    for(int i=0; i<input_shape[0]; i++){
        for(int j=0;j<input_shape[1];j++){
            std::cout << output[i*input_shape[1] + j ] << " ";
        }
        std::cout << std::endl;
    }


}