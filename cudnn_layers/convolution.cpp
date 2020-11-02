#include "convolution.h"
#include <iostream>
#include "../common.h"

Convolution::Convolution(int kernel_size[], int input_size[], cudnnHandle_t handle)
{
    type = CONV_LAYER;

    for(int i=0;i<3;i++)filter_shape[i] = kernel_size[i]; // HxWxOutputChannels
    for(int i=0;i<4;i++)input_shape[i] = input_size[i];  // BSXCXHXW

    int batch_size = input_shape[0];
    int input_channels = input_shape[1]; 
    int input_height = input_shape[2]; 
    int input_width = input_shape[3];
    int output_channels = kernel_size[2];
    int kernel_height = kernel_size[0];
    int kernel_width = kernel_size[1];
    
    // make input descriptor
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                /*format=*/CUDNN_TENSOR_NCHW,
                                /*dataType=*/CUDNN_DATA_FLOAT,
                                /*batch_size=*/batch_size,
                                /*channels=*/input_channels,
                                /*image_height=*/input_height,
                                /*image_width=*/input_width));
                    
    // make filter descriptor
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                /*dataType=*/CUDNN_DATA_FLOAT,
                                /*format=*/CUDNN_TENSOR_NCHW,
                                /*out_channels=*/output_channels,
                                /*in_channels=*/input_channels,
                                /*kernel_height=*/kernel_height,
                                /*kernel_width=*/kernel_width));

    // make convolution descriptor
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                    /*pad_height=*/0,
                                    /*pad_width=*/0,
                                    /*vertical_stride=*/1,
                                    /*horizontal_stride=*/1,
                                    /*dilation_height=*/1,
                                    /*dilation_width=*/1,
                                    /*mode=*/CUDNN_CROSS_CORRELATION,
                                    /*computeType=*/CUDNN_DATA_FLOAT));
    
    //get output shape 
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
                                        input_descriptor,
                                        kernel_descriptor,
                                        &output_shape[0],
                                        &output_shape[1],
                                        &output_shape[2],
                                        &output_shape[3]));
    
    // set output descriptor
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                            /*format=*/CUDNN_TENSOR_NCHW,
                            /*dataType=*/CUDNN_DATA_FLOAT,
                            /*batch_size=*/output_shape[0],
                            /*channels=*/output_shape[1],
                            /*image_height=*/output_shape[2],
                            /*image_width=*/output_shape[3]));

    // get fastest convolution algorithm
    checkCUDNN(
    cudnnGetConvolutionForwardAlgorithm(handle,
                                input_descriptor,
                                kernel_descriptor,
                                convolution_descriptor,
                                output_descriptor,
                                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                /*memoryLimitInBytes=*/0,
                                &convolution_algorithm));
    
    size_t forward_workspace_size, backward_filter_workspace_size, backward_data_workspace_size;
    
    //get forward workspace size
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(handle,
                                            input_descriptor,
                                            kernel_descriptor,
                                            convolution_descriptor,
                                            output_descriptor,
                                            convolution_algorithm,
                                            &forward_workspace_size));

    //get fastest backward algorithm for filter
    checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(
    handle, input_descriptor, output_descriptor, convolution_descriptor, kernel_descriptor,
    CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &filter_algorithm));

    //get workspace size for backward algorithm for filter
    checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    handle, input_descriptor, output_descriptor, convolution_descriptor, kernel_descriptor,
    filter_algorithm, &backward_filter_workspace_size));

    //get fastest backward algorithm for data
    checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(
    handle, kernel_descriptor, output_descriptor, convolution_descriptor, input_descriptor,
    CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &data_algorithm));

    //get workspace size for  backward algorithm for data
    checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
    handle, kernel_descriptor, output_descriptor, convolution_descriptor, input_descriptor,
    data_algorithm, &backward_data_workspace_size));

    workspace_size = std::max(forward_workspace_size, backward_filter_workspace_size);
    workspace_size = std::max(workspace_size, backward_data_workspace_size);
}

void Convolution::forward(float * input_activations)
{
    std::cout << "Convolution Forward Pass" << std::endl;
}

void Convolution::backward(float * output_gradients)
{
    std::cout << "Convolution Backward Pass" << std::endl;
}

int Convolution::get_workspace_size()
{
    return workspace_size;
}

void allocate_internal_memory()
{
    //allocate params memory 
  float* init_params = (float*) malloc(this->get_output_size());
  std::normal_distribution<float> distribution(MU,SIGMA);
  std::default_random_engine generator;

  int dim1 = filter_shape[1]*input_shape[1];
  int dim2 = filter_shape[0]*dim1;

  for(int ochannel = 0; ochannel < filter_shape[2]; ochannel++)
    for(int row=0;row<filter_shape[0];row++)
      for(int col=0;col<filter_shape[1];col++)
        for(int ichannel=0;ichannel < input_shape[1]; ichannel++)
          init_params[ochannel*dim2 + row*dim1 + col*input_shape[1] + ichannel] = distribution(generator);


  checkCUDA(cudaMemcpy(d_kernel,init_params,ochannels*ikernel_height*ikernel_width*ichannels*sizeof(float),cudaMemcpyHostToDevice));
}