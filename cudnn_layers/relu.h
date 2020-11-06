#ifndef relu_H_
#define relu_H_

#include "generic_layer.h"

class ReLU : public Layer 
{
    public:
        cudnnHandle_t handle; /*!< CUDNN Handle. */
        cudnnTensorDescriptor_t input_descriptor; /*!< cudnnTensor descriptor for input data to this layer. */
        cudnnTensorDescriptor_t output_descriptor; /*!< cudnnTensor descriptor for ouput data from this layer. */
        cudnnActivationDescriptor_t activation_descriptor; /*!< cudnn descriptor for activation operation. */
        ReLU(int input_size[], cudnnHandle_t handle);
        void forward(float * input_activations, float * output_activations);
        void backward(float * output_gradients, float * input_gradients, float * input_activations, float * output_activations=NULL);
        int get_workspace_size();
        void allocate_internal_memory();
        int get_param_size();
};

#endif