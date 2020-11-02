#include "cudnn_layers/generic_layer.h"
#include "cudnn_layers/convolution.h"
#include <iostream>

int main(int argc, const char* argv[])
{
    int filter_size[3] = {3, 3, 128};     //HWC
    int input_shape[4] = {64, 1, 28, 28}; //NCHW
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    Layer * conv_layer = new Convolution(filter_size, input_shape, cudnn);
    
    std::cout << conv_layer->get_type() << std::endl;
    int output_shape[4];
    conv_layer->get_output_shape(output_shape);

    std::cout << "Output Shape:\n" << std::endl;
    for(int i=0;i<4;i++)std::cout << output_shape[i] << " " ;
    std::cout << std::endl;

    std::cout << "Workspace Size: " << conv_layer->get_workspace_size() << std::endl;
    std::cout << "Input Size: " << conv_layer->get_input_size() << std::endl;
    std::cout << "Output Size: " << conv_layer->get_output_size() << std::endl;

    

}