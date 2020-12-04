#include "cudnn_layers/generic_layer.h"
#include "cudnn_layers/convolution.h"
#include "cudnn_layers/fc.h"
#include "cudnn_layers/relu.h"
#include "common.h"

#include "mpi.h"
#include "cuda_runtime.h"
#include "nccl.h"

#include <unistd.h>
#include <iostream>
#include <vector>
#include <sstream>

int ncclStreamSynchronize(cudaStream_t stream, ncclComm_t comm) {
  cudaError_t cudaErr;
  ncclResult_t ncclErr, ncclAsyncErr;
  while (1) {
   cudaErr = cudaStreamQuery(stream);
   if (cudaErr == cudaSuccess)
     return 0;

   if (cudaErr != cudaErrorNotReady) {
     printf("CUDA Error : cudaStreamQuery returned %d\n", cudaErr);
     return 1;
   }

   ncclErr = ncclCommGetAsyncError(comm, &ncclAsyncErr);
   if (ncclErr != ncclSuccess) {
     printf("NCCL Error : ncclCommGetAsyncError returned %d\n", ncclErr);
     return 1;
   }

   if (ncclAsyncErr != ncclSuccess) {
     // An asynchronous error happened. Stop the operation and destroy
     // the communicator
     ncclErr = ncclCommAbort(comm);
     if (ncclErr != ncclSuccess)
       printf("NCCL Error : ncclCommDestroy returned %d\n", ncclErr);
     // Caller may abort or try to re-create a new communicator.
     return 2;
   }

  }
}

static uint64_t getHostHash(const char* string) {
    // Based on DJB2, result = result * 33 + char
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++){
        result = ((result << 5) + result) + string[c];
    }
    return result;
}

static void getHostName(char* hostname, int maxlen) {
    gethostname(hostname, maxlen);
    for (int i=0; i< maxlen; i++) {
        if (hostname[i] == '.') {
            hostname[i] = '\0';
            return;
        }
    }
}

//calculating local_rank which is used in selecting a GPU
static int get_local_rank(int my_rank, int n_ranks) {
    int local_rank = 0;
    uint64_t hostHashs[n_ranks];
    char hostname[1024];
    getHostName(hostname, 1024);
    hostHashs[my_rank] = getHostHash(hostname);
    MPICHECK(MPI_Allgather(
        MPI_IN_PLACE,
        0,
        MPI_DATATYPE_NULL,
        hostHashs,
        sizeof(uint64_t),
        MPI_BYTE,
        MPI_COMM_WORLD
    ));

    for (int p=0; p<n_ranks; p++) {
        if (p == my_rank) break;
        if (hostHashs[p] == hostHashs[my_rank]) local_rank++;
    }

    return local_rank;
}

class NN
{
    public:
        int num_layers;
        Layer ** network;
        
        NN(std::vector<std::string> nn_config,int input_shape[], cudnnHandle_t cudnn, cublasHandle_t cublas)
        {
            num_layers = nn_config.size();
            network = (Layer**)malloc(num_layers*sizeof(Layer*));
            for(int i=0; i<num_layers; i++)
            {
                std::istringstream iss (nn_config[i]);
                std::string layer_type;
                iss >> layer_type;
                std::cout << layer_type << " ";
                int ul=0, dim[3];
                if(layer_type == "conv2d") ul = 3;
                else if(layer_type == "fc") ul = 1;
                for(int j=0;j<ul;j++){
                    iss >> dim[j];
                    std::cout << dim[j] << " ";
                }
                if(layer_type == "conv2d")
                    network[i] = new Convolution(dim, input_shape, cudnn);
                else if (layer_type== "fc")
                    network[i] = new FC(dim[0], input_shape, cublas);
                else 
                    network[i] = new ReLU(input_shape, cudnn);
                
                network[i]->get_output_shape(input_shape);
                std::cout << std::endl << "output shape ";
                for(int j=0;j<4;j++)std::cout << input_shape[j] << " ";
                std::cout << std::endl;
                

            }
        }

        int get_num_layers(){
            return num_layers;
        }

        Layer ** get_network_obj(){
            return network;
        }
};


int main(int argc, char* argv[])
{
    int my_rank, n_ranks, local_rank = 0;
    float min_time, max_time, sum_time;
    
    //initializing MPI
    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &my_rank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &n_ranks));
 

    // Assume each rank gets one gpu for now
    local_rank = get_local_rank(my_rank, n_ranks);
    std::cout << "My local rank : " << local_rank << std::endl;
    
    
    checkCUDA(cudaSetDevice(local_rank));
    
    
    ncclUniqueId id;
    ncclComm_t comm;
    //generating NCCL unique ID at one process and broadcasting it to all
    if (my_rank == 0) ncclGetUniqueId(&id);
    MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
    NCCLCHECK(ncclCommInitRank(&comm, n_ranks, id, my_rank));
    
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    cublasHandle_t cublas;
    cublasCreate(&cublas);
    cudaStream_t nccl_comm_stream, kernel_exec_stream;
    checkCUDA(cudaStreamCreate(&nccl_comm_stream));
    checkCUDA(cudaStreamCreate(&kernel_exec_stream));
    checkCUDNN(cudnnSetStream(cudnn, kernel_exec_stream));
    cublasSetStream(cublas, kernel_exec_stream);

    //Create a Simple LeNet
    int input_shape[4] = {64, 1, 100, 100};
    NN * neural_network = new NN({"conv2d 3 3 32",
                                  "ReLU",
                                  "conv2d 3 3 64",
                                  "ReLU",
                                  "conv2d 3 3 10",
                                  "ReLU",
                                  "conv2d 3 3 10","ReLU",
				  "fc 50","ReLU", "fc 10"
                                 }, 
                                     input_shape, cudnn, cublas);

    Layer ** network = neural_network->get_network_obj();
    int num_layers = neural_network->get_num_layers();
    
    
    int device;
    checkCUDA(cudaGetDevice(&device)); 	
    std::cout << "My device is : "<< device << std::endl;
    
    
    //Do a forward Pass
    //Step 1 - Copy batch to GPU - Here we will generate random batch
    int input_size = network[0]->get_input_size();
    float *d_batch, *d_grad_batch, *batch;
    checkCUDA(cudaMalloc(&d_batch, input_size));
    checkCUDA(cudaMalloc(&d_grad_batch, input_size));
    
    batch = (float*)malloc(input_size);
    std::normal_distribution<float> distribution(MU,SIGMA);
    std::default_random_engine generator;
    for(int i=0; i<input_size/sizeof(float); i++)batch[i] = distribution(generator);
    checkCUDA(cudaMemcpy(d_batch, batch, input_size, cudaMemcpyHostToDevice));

    //Step 2 - Allocate internal memory for all layers
    for(int i=0; i<num_layers; i++)network[i]->allocate_internal_memory();

    //Step 3 - Allocate output activation buffers for each layer
    float *output_activations[num_layers], *grad_output_activations[num_layers];
    for(int i=0; i<num_layers; i++)
    {
        int output_size = network[i]->get_output_size();
        checkCUDA(cudaMalloc(&output_activations[i], output_size));
        checkCUDA(cudaMalloc(&grad_output_activations[i], output_size));
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, kernel_exec_stream);
    for(int X=0;X<N_BATCHES;X++)
    {
        network[0]->forward(d_batch, output_activations[0]);
        //std::cout <<"Local Rank "<<local_rank <<" " <<"FW Layer 0" << std::endl;
        checkCUDA(cudaStreamSynchronize(kernel_exec_stream));
        for(int i=1;i<num_layers;i++)
        {
            //MPI_Barrier(MPI_COMM_WORLD);
            network[i]->forward(output_activations[i-1], output_activations[i]);
            //std::cout <<"Local Rank "<<local_rank <<" " <<"FW Layer " << i << std::endl; 
            //checkCUDA(cudaStreamSynchronize(kernel_exec_stream));
        }
        std::cout <<"Local Rank "<<local_rank <<" " <<"FW Pass Done"<< std::endl;
        
        //Step 5 - Print output of final layer
        int output_size = network[num_layers-1]->get_output_size();

        //Step 6 - Use random gradient for output right now
        float * grad_output = (float*) malloc(output_size);
        for(int i=0; i<output_size/sizeof(float); i++)
            grad_output[i] = distribution(generator);
        checkCUDA(cudaMemcpy(grad_output_activations[num_layers-1], grad_output, output_size, cudaMemcpyHostToDevice));

        //Step 7 - Do backward Pass 
        for(int i=num_layers-1; i>0; i--)
        {
            network[i]->backward(
                grad_output_activations[i],
                grad_output_activations[i-1],
                output_activations[i-1],
                output_activations[i]
            );
            
            // std::cout <<"Local Rank "<<local_rank <<" " <<"BW Layer " << i << std::endl;
            if(network[i]->get_param_size()>0){ 
                checkCUDA(cudaStreamSynchronize(kernel_exec_stream));
                NCCLCHECK(ncclAllReduce(network[i]->params_gradients, network[i]->params_gradients_nccl, network[i]->get_param_size(), ncclFloat, ncclSum, comm, nccl_comm_stream));
            }
        }
        // first layer is special
        network[0]->backward(
            grad_output_activations[0],
            d_grad_batch,
            d_batch,
            output_activations[0]
        );
        // std::cout <<"Local Rank "<<local_rank <<" " <<"BW Layer " << 0 << std::endl; 
        
        if(network[0]->get_param_size()>0){
            checkCUDA(cudaStreamSynchronize(kernel_exec_stream));
            NCCLCHECK(ncclAllReduce(network[0]->params_gradients, network[0]->params_gradients_nccl, network[0]->get_param_size(), ncclFloat, ncclSum, comm ,nccl_comm_stream));
        }
        
        int a = ncclStreamSynchronize(nccl_comm_stream, comm);

        if(a!=0)break;
        //checkCUDA(cudaStreamSynchronize(nccl_comm_stream));
        std::cout <<"Local Rank "<<local_rank <<" " <<"BW Pass Done"<< std::endl;
    } 
    cudaEventRecord(stop, kernel_exec_stream);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
	//reduce operations to print min, max, and avg time.
	MPICHECK(MPI_Reduce(&milliseconds, &sum_time, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD));
	MPICHECK(MPI_Reduce(&milliseconds, &min_time, 1, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD));
	MPICHECK(MPI_Reduce(&milliseconds, &max_time, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD));

    if(my_rank==0)
        std::cout << "TIME: Min: " << min_time/1000 << " s " << "Avg: " << (sum_time/n_ranks)/1000 << " s " << "Max: " << max_time/1000 << " s" << std::endl;

    ncclCommDestroy(comm);
    MPICHECK(MPI_Finalize());
    checkCUDA(cudaStreamDestroy(kernel_exec_stream));
    checkCUDA(cudaStreamDestroy(nccl_comm_stream));
    checkCUDNN(cudnnDestroy(cudnn));
    cublasDestroy(cublas);


    std::cout << "Rank " << my_rank << " done" << std::endl;

}
