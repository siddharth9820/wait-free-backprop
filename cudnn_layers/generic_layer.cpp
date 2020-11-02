#include "generic_layer.h"

void Layer::get_input_shape(int shape[])
{
    for(int i=0;i<4;i++)shape[i] = input_shape[i];
}

void Layer::get_output_shape(int shape[])
{
    for(int i=0;i<4;i++)shape[i] = output_shape[i];
}

int Layer::get_type()
{
    return type;
}

int Layer::get_input_size()
{
    int ip_size = 1;
    for(int i=0;i<4;i++) ip_size *= input_shape[i];
    return ip_size * sizeof(float);
}

int Layer::get_output_size()
{
    int op_size = 1;
    for(int i=0;i<4;i++) op_size *= output_shape[i];
    return op_size * sizeof(float);
}


