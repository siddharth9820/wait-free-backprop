#ifndef FC_H_
#define FC_H_

#include "generic_layer.h"

class FC : public Layer 
{
    public:
        cublasHandle_t handle; 
        FC(int output_size, int input_size[], cublasHandle_t handle);
        void forward(float * input_activations, float * output_activations);
        void backward(float * output_gradients, float * input_gradients, float * input_activations, float * output_activations);
        int get_workspace_size();
        void allocate_internal_memory();
        int get_param_size();
};

#endif