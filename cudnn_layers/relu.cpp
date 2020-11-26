#include "relu.h"

ReLU::ReLU(int input_size[], cudnnHandle_t handle)
{
    type = FC_LAYER;
    for(int i=0;i<4;i++) input_shape[i] = output_shape[i] = input_size[i];
    
    this->handle = handle;
    cudnnCreateTensorDescriptor(&input_descriptor);
    cudnnCreateTensorDescriptor(&output_descriptor);
    cudnnCreateActivationDescriptor(&activation_descriptor);
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,input_shape[0],input_shape[1],input_shape[2],input_shape[3]));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,input_shape[0],input_shape[1],input_shape[2],input_shape[3]));
    checkCUDNN(cudnnSetActivationDescriptor(activation_descriptor,CUDNN_ACTIVATION_RELU,CUDNN_PROPAGATE_NAN,0));
    
}

int ReLU::get_workspace_size()
{
    return 0;
}

int ReLU::get_param_size()
{
    return 0;
}

void ReLU::allocate_internal_memory()
{
    return;   
}

void ReLU::forward(float * input_activations, float * output_activations)
{
    float alpha = 1, beta=0;
    checkCUDNN(cudnnActivationForward(handle,
                activation_descriptor,
                &alpha,
                input_descriptor,
                input_activations,
                &beta,
                output_descriptor,
                output_activations));
}

void ReLU::backward(float * output_gradients, float * input_gradients, float * input_activations, float * output_activations)
{
    float alpha = 1.0;
    float beta = 0.0;
    checkCUDNN(cudnnActivationBackward(handle,
      activation_descriptor,
      &alpha,
      output_descriptor,
      output_activations,
      output_descriptor,
      output_gradients,
      input_descriptor,
      input_activations,
      &beta,
      input_descriptor,
      input_gradients
    ));

}
