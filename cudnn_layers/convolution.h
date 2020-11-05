#ifndef CONVOLUTION_H_
#define CONVOLUTION_H_

#include "generic_layer.h"

class Convolution : public Layer 
{
    public:
        cudnnHandle_t handle; /*!< CUDNN Handle. */
        cudnnTensorDescriptor_t input_descriptor;
        cudnnFilterDescriptor_t kernel_descriptor; /*!< cudnnTensor descriptor for kernel of this layer. */
        cudnnConvolutionDescriptor_t convolution_descriptor; /*!< cudnn convolution operation descriptor. */
        cudnnTensorDescriptor_t output_descriptor; /*!< cudnnTensor descriptor for output data from this layer. */
        cudnnTensorDescriptor_t filter_derivative_descriptor; /*!< cudnnTensor descriptor for filter gradients of this layer. */
        cudnnConvolutionFwdAlgo_t convolution_algorithm; /*!< cudnn descriptor for convolution forward pass algorithm. */
        cudnnConvolutionBwdFilterAlgo_t filter_algorithm; /*!< cudnn descriptor for backward pass filter algorithm. */
        cudnnConvolutionBwdDataAlgo_t data_algorithm; /*!< cudnn descriptor for backward pass data algorithm. */
        int filter_shape[3];
        size_t workspace_size;
        

        Convolution(int kernel_size[], int input_size[], cudnnHandle_t handle);
        void forward(float * input_activations, float * output_activations);
        void backward(float * output_gradients, float * input_gradients, float * input_activations);
        int get_workspace_size();
        void allocate_internal_memory();
        int get_param_size();
};

#endif