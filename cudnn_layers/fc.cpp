#include "fc.h"

const char* cublasGetErrorString(cublasStatus_t status)
{
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "unknown error";
}


FC::FC(int output_size, int input_size[], cublasHandle_t handle)
{
    type = ReLU_LAYER;
    input_shape[0] = input_size[0];
    input_shape[1] = input_size[1] * input_size[2] * input_size[3];
    input_shape[2] = input_shape[3] = 1;
    output_shape[0] = input_shape[0];
    output_shape[1] = output_size;
    output_shape[2] = 1;
    output_shape[3] = 1;
    this->handle = handle;
}

int FC::get_workspace_size()
{
    return 0;
}

int FC::get_param_size()
{
    return input_shape[1] * output_shape[1] * sizeof(float);
}

void FC::allocate_internal_memory()
{
    int param_size = this->get_param_size();
    std::cout << "Parameter memory = " << param_size << " bytes" << std::endl;
    float* cpu_params = (float*) malloc(param_size);
    
    std::normal_distribution<float> distribution(MU,SIGMA);
    std::default_random_engine generator;

    for(int i=0; i<param_size/sizeof(float); i++)
        cpu_params[i] = distribution(generator);

    checkCUDA(cudaMalloc(&params, param_size));
    checkCUDA(cudaMalloc(&params_gradients, param_size));
    checkCUDA(cudaMemcpy(params, cpu_params, param_size, cudaMemcpyHostToDevice));
    
}

void FC::forward(float * input_activations, float * output_activations)
{
    int output_height = output_shape[1];
    int input_height = input_shape[1];
    int batch_size = output_shape[0];
    float alpha = 1, beta=0;

    checkCUBLAS(cublasSgemm(handle,
                CUBLAS_OP_N, 
                CUBLAS_OP_N, 
                output_height,/*N*/
                batch_size,/*M*/
                input_height,/*K*/
                &alpha,
                params, //B
                output_height, //N
                input_activations, //A
                input_height, //K
                &beta,
                output_activations,//C
                output_height //K
            ));
}

void FC::backward(float * output_gradients, float * input_gradients, float * input_activations, float * output_activations)
{
    int output_height = output_shape[1];
    int input_height = input_shape[1];
    int batch_size = output_shape[0];
    float alpha = 1, beta=0;

    checkCUBLAS(cublasSgemm(handle,
                CUBLAS_OP_N, //info for B, use CUBLAS_OP_T if you want to use BT
                CUBLAS_OP_T, //info for A, use CUBLAS_OP_T if you want to use AT
                output_height,/*N*/
                input_height,/*M*/
                batch_size,/*K*/
                &alpha,
                output_gradients, //B
                output_height, //N
                input_activations, //A
                input_height, //K
                &beta,
                params_gradients,//C
                output_height //K
            ));

  checkCUBLAS(cublasSgemm(handle,
              CUBLAS_OP_T, //info for B, use CUBLAS_OP_T if you want to use BT
              CUBLAS_OP_N, //info for A, use CUBLAS_OP_T if you want to use AT
              input_height,/*N*/
              batch_size,/*M*/
              output_height,/*K*/
              &alpha,
              params, //B
              output_height, //N
              output_gradients, //A
              output_height, //K
              &beta,
              input_gradients,//C
              input_height //K
            ));

}