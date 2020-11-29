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
#include <string>

#include <time.h>
#include <sys/time.h>

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}


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
        float ** output_activations;
        float ** grad_output_activations;
        cudaStream_t execute_stream, comm_stream;
        ncclComm_t comm;
        cudaEvent_t * events;

        NN(std::vector<std::string> nn_config,int input_shape[], cudnnHandle_t cudnn, cublasHandle_t cublas, cudaStream_t nccl_comm_stream, cudaStream_t kernel_exec_stream, ncclComm_t comm)
        {
            
            execute_stream = kernel_exec_stream;
            comm_stream = nccl_comm_stream;
            this->comm = comm;
            num_layers = nn_config.size();
            events = (cudaEvent_t *)malloc(num_layers * sizeof(cudaEvent_t)); 
            network = (Layer**)malloc(num_layers*sizeof(Layer*));
            for(int i=0; i<num_layers; i++)cudaEventCreate(&events[i]);
            
            for(int i=0; i<num_layers; i++)
            {
                std::istringstream iss (nn_config[i]);
                std::string layer_type;
                iss >> layer_type;
                //std::cout << layer_type << " ";
                int ul=0, dim[3];
                if(layer_type == "conv2d") ul = 3;
                else if(layer_type == "fc") ul = 1;
                for(int j=0;j<ul;j++){
                    iss >> dim[j];
                    //std::cout << dim[j] << " ";
                }
                if(layer_type == "conv2d")
                    network[i] = new Convolution(dim, input_shape, cudnn);
                else if (layer_type== "fc")
                    network[i] = new FC(dim[0], input_shape, cublas);
                else 
                    network[i] = new ReLU(input_shape, cudnn);
                
                network[i]->get_output_shape(input_shape);
                //std::cout << std::endl << "output shape ";
                //for(int j=0;j<4;j++)std::cout << input_shape[j] << " ";
                //std::cout << std::endl;
            }
        }

        int get_num_layers(){
            return num_layers;
        }

        Layer ** get_network_obj(){
            return network;
        }

        int get_output_size(){
            return network[num_layers-1]->get_output_size();
        }

        int get_input_size(){
            return network[0]->get_input_size();
        }

        void get_output_shape(int shape[]){
            network[num_layers-1]->get_output_shape(shape);
        }

        void allocate_memory(){
            output_activations = (float**)malloc(num_layers*sizeof(float*));
            grad_output_activations = (float**)malloc(num_layers*sizeof(float*));
            for(int i=0; i<num_layers; i++)network[i]->allocate_internal_memory();
            for(int i=0; i<num_layers; i++){
                int output_size = network[i]->get_output_size();
                checkCUDA(cudaMalloc(&output_activations[i], output_size));
                if(i < num_layers - 1)
                    checkCUDA(cudaMalloc(&grad_output_activations[i], output_size));
            }
        }

        void forward(float * d_batch){
            network[0]->forward(d_batch, output_activations[0]);
            for(int i=1; i<num_layers; i++)network[i]->forward(output_activations[i-1], output_activations[i]);
        }

        void backward_wfbp(float * d_grad_output, float * d_batch, float * d_grad_batch){
            grad_output_activations[num_layers-1] = d_grad_output;
            
            for(int i=num_layers-1; i>0; i--){
                network[i]->backward(
                    grad_output_activations[i],
                    grad_output_activations[i-1],
                    output_activations[i-1],
                    output_activations[i]
                );
                cudaEventRecord(events[i], execute_stream);
            }
            network[0]->backward(
                    grad_output_activations[0],
                    d_grad_batch,
                    d_batch,
                    output_activations[0]
            );
            cudaEventRecord(events[0], execute_stream);

            for(int i=num_layers-1; i>=0; i--){
                if(network[i]->get_param_size() > 0){
                cudaStreamWaitEvent(comm_stream, events[i], 0);
                   NCCLCHECK(ncclAllReduce(network[i]->params_gradients, network[i]->params_gradients, network[i]->get_param_size(), ncclFloat, ncclSum, comm, comm_stream));
                }
            }
        
        }

        void backward_vanilla(float * d_grad_output, float * d_batch, float * d_grad_batch){
            grad_output_activations[num_layers-1] = d_grad_output;
            for(int i=num_layers-1; i>0; i--)
            {
                network[i]->backward(
                    grad_output_activations[i],
                    grad_output_activations[i-1],
                    output_activations[i-1],
                    output_activations[i]
                ); 
            }
            network[0]->backward(
                    grad_output_activations[0],
                    d_grad_batch,
                    d_batch,
                    output_activations[0]
            );

            checkCUDA(cudaStreamSynchronize(execute_stream));
            
            for(int i=num_layers-1; i>=0; i--){
                if(network[i]->get_param_size() > 0){
                   NCCLCHECK(ncclAllReduce(network[i]->params_gradients, network[i]->params_gradients_nccl, network[i]->get_param_size(), ncclFloat, ncclSum, comm, comm_stream));
                }
            }
        }
};


int main(int argc, char* argv[])
{
    int my_rank, n_ranks, local_rank = 0;
    std::string nepochs(argv[1]);
    int num_epochs = std::stoi(nepochs);
    std::string mode(argv[2]);

    //initializing MPI
    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &my_rank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &n_ranks));
 
   
    // Assume each rank gets one gpu for now
    local_rank = get_local_rank(my_rank, n_ranks);
    std::cout << "Global Rank " << my_rank <<" Local Rank " << local_rank << std::endl;
    
    
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
    int input_shape[4] = {64, 3, 50, 50};
    NN * neural_network = new NN({"conv2d 3 3 64",
                                  "ReLU",
                                  "conv2d 3 3 128",
                                  "ReLU",
                                 }, 
                                input_shape, cudnn, cublas, nccl_comm_stream, kernel_exec_stream, comm);
    
    //Do a forward Pass
    //Step 1 - Copy batch to GPU - Here we will generate random batch
    int input_size = neural_network->get_input_size();
    int output_size = neural_network->get_output_size();
    int output_shape[4];
    neural_network->get_output_shape(output_shape);
    if(my_rank == 0){
        std::cout << "The output shape is : " << output_shape[0] << " " << output_shape[1] << " " << output_shape[2] << " " << output_shape[3] << std::endl;
        if(mode=="vanilla") std::cout << "Doing Vanilla BackPropogation" << std::endl;
        else if(mode == "wfbp")std::cout << "Doing Wait-Free BackPropogation" << std::endl;
    }

    float *d_batch, *d_grad_batch, *batch, *d_grad_output;
    checkCUDA(cudaMalloc(&d_batch, input_size));
    checkCUDA(cudaMalloc(&d_grad_batch, input_size));
    checkCUDA(cudaMalloc(&d_grad_output, output_size));
    batch = (float*)malloc(input_size);
    float * grad_output = (float*) malloc(output_size);

    std::normal_distribution<float> distribution(MU,SIGMA);
    std::default_random_engine generator;
    
    neural_network->allocate_memory();

    double start = get_wall_time();
    for(int X=0;X<num_epochs;X++)
    {
        if(my_rank == 0)
            std::cout << "Epoch number "<<X+1 <<std::endl;
        for(int i=0; i<input_size/sizeof(float); i++)batch[i] = distribution(generator);
        checkCUDA(cudaMemcpy(d_batch, batch, input_size, cudaMemcpyHostToDevice));
        
        neural_network->forward(d_batch);        
        
        for(int i=0; i<output_size/sizeof(float); i++)
            grad_output[i] = distribution(generator);
        
        checkCUDA(cudaMemcpy(d_grad_output, grad_output, output_size, cudaMemcpyHostToDevice));

        if (mode == "vanilla")
            neural_network->backward_vanilla(d_grad_output, d_batch, d_grad_batch);
        else if(mode == "wfbp")
            neural_network->backward_wfbp(d_grad_output, d_batch, d_grad_batch); 

        checkCUDA(cudaStreamSynchronize(nccl_comm_stream));
    } 
    double end = get_wall_time();
    ncclCommDestroy(comm);
    MPICHECK(MPI_Finalize());
    checkCUDA(cudaStreamDestroy(kernel_exec_stream));
    checkCUDA(cudaStreamDestroy(nccl_comm_stream));
    checkCUDNN(cudnnDestroy(cudnn));
    cublasDestroy(cublas);


    std::cout << "Rank " << my_rank << " done" << " Time taken is :" << (end-start)/num_epochs <<" seconds" << std::endl;

}
