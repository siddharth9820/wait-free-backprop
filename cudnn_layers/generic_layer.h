#ifndef GENERIC_LAYERS_H_
#define GENERIC_LAYERS_H_

#define CONV_LAYER 0
#define FC_LAYER 1
#define ReLU_LAYER 2

#include "../common.h"

class Layer
{   
    public:
        int input_shape[4];
        int output_shape[4];
        int type;
        float * workspace, * params, *params_gradients, *params_gradients_nccl;
        virtual void forward(float * input_activations, float * output_activations)=0;
        virtual void backward(float * output_gradients, float * input_gradients, float * input_activations, float * output_activations)=0;
        void get_input_shape(int shape[]);
        void get_output_shape(int shape[]);
        int get_type();
        int get_input_size();
        int get_output_size();
        virtual int get_workspace_size()=0;
        virtual void allocate_internal_memory()=0;
        virtual int get_param_size()=0;
};

#endif
