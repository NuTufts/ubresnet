import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F

from dummymodel import Dummytest

def main():
    print(Dummytest)
    input = torch.randn(1, 1, 28, 28)
    net = Dummytest()


    net.pool1.register_forward_hook(hook)

    out = net(input)
    print(out.size())

    pred = net.forward(input)
    print "Right after hook"
    print(outputs.sum().item())
    # pool1_tensor = outputs
    # print "pool1tensor_shape: ", pool1_tensor.shape
    # print "pool1tensor_sum: ", pool1_tensor.sum().item()


    print "FIN"

# -----------------Hooking functions--------------------------------

# global outputs
# outputs = torch.tensor(torch.zeros(1,1))

def hook(module, input, output):
    global outputs
    outputs = output.data
    print(output.data.sum().item(), outputs.data.sum().item())

def printnorm(self, input, output):
   # input is a tuple of packed inputs
   # output is a Tensor. output.data is the Tensor we are interested
   print('Inside ' + self.__class__.__name__ + ' forward')
   print('')
   print('input: ', type(input))
   print('input[0]: ', type(input[0]))
   print('output: ', type(output))
   print('')
   print('input size:', input[0].size())
   print('output size:', output.data.size())
   print('output norm:', output.data.norm())
# -------------------------------------------------------------------

if __name__ == '__main__':
    #dump_lr_schedule(1.0e-2, 4000)
    main()
