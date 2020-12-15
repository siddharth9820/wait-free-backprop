import os 
import torch 
import torch.nn as nn 
import torch.optim as optim 

from torch.nn.parallel import DistributedDataParallel as DDP 
import argparse
import torch.distributed as dist
import time

parser = argparse.ArgumentParser(description='Torch distributed data parallel')

parser.add_argument('--local_rank', default=0, type=int, help='number of distributed processes')
parser.add_argument('--dist_backend', default='gloo', type=str, help='distributed backend')
parser.add_argument('--cuda', action='store_true', default=False, help='enables GPU usage')
parser.add_argument('--batches', type=int, default=192, help='number of batches')
parser.add_argument('--weak-scale', action='store_true', default=False, help='use weak scaling')
parser.add_argument('--batch_size', type=int, default=64, help='batch_size')

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=10, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=10, out_channels=1, kernel_size=3)
        self.fc1 = nn.Linear(92*92, 50)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.fc1(x.view(x.shape[0],-1))
        x = self.relu(x)
        x = self.fc2(x)
        return x


def print_status(msg):
    print("RANK {} : {}".format(dist.get_rank(), msg))

def main():
    args = parser.parse_args()
    if not args.cuda:
        args.dist_backend = 'gloo' # nccl doesn't work on CPUs

    dist.init_process_group(backend=args.dist_backend, init_method='env://')
    model = Model() 
    if args.cuda:
        print_status("Using GPU")
        torch.cuda.set_device(args.local_rank)
        model.cuda()
    else:
        print_status("Using CPU")
    
    print_status("initialising DDP model")
    if args.cuda:
        ddp_model = DDP(model, device_ids=[torch.cuda.current_device()])
    else:
        ddp_model = DDP(model)
    
    num_batches = args.batches 
    if not args.weak_scale:
        print_status("Strong scaling")
        num_batches =  num_batches // dist.get_world_size() 
    batch_size = args.batch_size 
    start_time = time.time()
    
    for _ in range(num_batches):
        # create random batch 
        x = torch.randn(batch_size, 1, 100, 100)
        if args.cuda:
            x.cuda() 
        y = ddp_model(x)
        rand_grad = torch.randn_like(y)
        y.backward(rand_grad)

    end_time = time.time() 
    avg_time_tensor = torch.FloatTensor([end_time - start_time])
    min_time_tensor = torch.FloatTensor([end_time - start_time])
    max_time_tensor = torch.FloatTensor([end_time - start_time])
    
    if args.cuda:
        avg_time_tensor = avg_time_tensor.cuda()
        min_time_tensor = min_time_tensor.cuda()
        max_time_tensor = max_time_tensor.cuda()

    dist.reduce(avg_time_tensor, 0, dist.reduce_op.SUM)
    dist.reduce(min_time_tensor, 0, dist.reduce_op.MIN)
    dist.reduce(max_time_tensor, 0, dist.reduce_op.MAX)

    avg_time_tensor /= dist.get_world_size()

    time_min, time_avg, time_max = min_time_tensor.item(), avg_time_tensor.item(), max_time_tensor.item()

    if dist.get_rank() == 0:
        print_status("Time : Min {} Avg {} Max {}".format(time_min, time_avg, time_max))


if __name__ == "__main__":
    print("entering main")
    main()