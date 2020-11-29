#include "convolution.h"


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
    this->handle = handle;

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

    //get fastest convolution algorithm
    checkCUDNN(
    cudnnGetConvolutionForwardAlgorithm(handle,
                                input_descriptor,
                                kernel_descriptor,
                                convolution_descriptor,
                                output_descriptor,
                                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                /*memoryLimitInBytes=*/0,
                                &convolution_algorithm));
    

    //convolution_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

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

int Convolution::get_workspace_size()
{
    return workspace_size;
}

int Convolution::get_param_size()
{
    return input_shape[1] * filter_shape[0] * filter_shape[1] * filter_shape[2] * sizeof(float);
}

void Convolution::allocate_internal_memory()
{
    //allocate params memory 
    int filter_size = this->get_param_size();
    //std::cout << "Parameter memory = " << filter_size << " bytes" << std::endl;
    float* cpu_params = (float*) malloc(filter_size);
    
    std::normal_distribution<float> distribution(MU,SIGMA);
    std::default_random_engine generator;


    for(int i=0; i<filter_size/sizeof(float); i++)
        cpu_params[i] = distribution(generator);

    checkCUDA(cudaMalloc(&params, filter_size));
    checkCUDA(cudaMemcpy(params,cpu_params,filter_size,cudaMemcpyHostToDevice));
    checkCUDA(cudaMalloc(&params_gradients, filter_size));
    checkCUDA(cudaMalloc(&params_gradients_nccl, filter_size));
    //allocate worksapce memory
    checkCUDA(cudaMalloc(&workspace, workspace_size));

}

void Convolution::forward(float * input_activations, float * output_activations)
{
    float alpha=1, beta=0;
    checkCUDNN(cudnnConvolutionForward(handle,
                                       &alpha,
                                       input_descriptor,
                                       input_activations,
                                       kernel_descriptor,
                                       params,
                                       convolution_descriptor,
                                       convolution_algorithm,
                                       workspace,
                                       workspace_size,
                                       &beta,
                                       output_descriptor,
                                       output_activations));
}

void Convolution::backward(float * output_gradients, float * input_gradients, float * input_activations, float * output_activations)
{
    float alpha=1, beta=0;
    checkCUDNN(cudnnConvolutionBackwardData(
    handle,
    &alpha,
    kernel_descriptor,
    params,
    output_descriptor,
    output_gradients,
    convolution_descriptor,
    data_algorithm,
    workspace,
    workspace_size,
    &beta,
    input_descriptor,
    input_gradients
    ));

    checkCUDNN(cudnnConvolutionBackwardFilter(
        handle,
        &alpha,
        input_descriptor,
        input_activations,
        output_descriptor,
        output_gradients,
        convolution_descriptor,
        filter_algorithm,
        workspace,
        workspace_size,
        &beta,
        kernel_descriptor,
        params_gradients
    ));

}
