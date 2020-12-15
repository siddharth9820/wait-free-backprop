#include "dataloader.h"


MNIST_loader::MNIST_loader(int batch_size, bool show_samples)
{
    train = new mnist_loader("mnist-loader/dataset/train-images-idx3-ubyte",
      "mnist-loader/dataset/train-labels-idx1-ubyte", 0);
    test = new mnist_loader("mnist-loader/dataset/t10k-images-idx3-ubyte",
                "mnist-loader/dataset/t10k-labels-idx1-ubyte", 0);

    rows = train->rows();
    cols = train->cols();
    
    train_size = train->size();
    test_size = test->size();
    this->batch_size = batch_size;

    if (show_samples){
      std::cout << "visualising 2 random images out of " << train_size << " images" << std::endl;
      for(int i=0 ; i<2; i++){
        int index = rand() % train_size;
        int label = train->labels(index);
        std::cout << "Category - " << label << std::endl;
        visualize(train->images(index));
      }
    }
    curr = 0;
}

void MNIST_loader::get_input_shape(int shape[]){
  shape[0] = batch_size;
  shape[1] = 1;
  shape[2] = rows;
  shape[3] = cols;
}

void MNIST_loader::init_memory(float ** batch, int **labels)
{
  (*batch) = (float *)malloc(batch_size * rows * cols * sizeof(float));
  (*labels) = (int *)malloc(batch_size * sizeof(int));
}

void MNIST_loader::get_next_batch(float * batch, int * labels)
{
    for(int i=0;i<batch_size;i++){
        std::vector<double> image = train->images(curr);
        labels[i] = train->labels(curr);
        int sz = image.size();
        for(int j=0; j<sz; j++)batch[i*sz+j] = image[j];
        curr = (curr + 1) % train_size;
    }
}

void MNIST_loader::visualize(float * batch, int * labels, int num)
{
  for(int i=0; i<num; i++)
  {
    int index = rand() % batch_size;;
    std::cout << "Label : " << labels[index] << std::endl;
    for (int y=0; y<rows; ++y) {
      for (int x=0; x<cols; ++x){
        std::cout << ((batch[index*cols*rows + y*cols+x] == 0.0)? ' ':'*');
      }
      std::cout << std::endl;
    }
  }
}

void MNIST_loader::visualize(std::vector<double> image)
{
  std::cout << "inside visualization" << std::endl;
  for (int y=0; y<rows; ++y) {
    for (int x=0; x<cols; ++x) {
      std::cout << ((image[y*cols+x] == 0.0)? ' ' : '*');
    }
    std::cout << std::endl;
  }
}

