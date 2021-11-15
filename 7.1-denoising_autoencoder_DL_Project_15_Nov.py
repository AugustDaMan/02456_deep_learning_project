#!/usr/bin/env python
# coding: utf-8

# ## IImport libraries

# In[1]:


#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch
cuda = torch.cuda.is_available()

try:
    from plotting_CIFAR10 import plot_autoencoder_stats
except Exception as ex:
    print(f"If using Colab, you may need to upload `plotting.py`.           \nIn the left pannel, click `Files > upload to session storage` and select the file `plotting.py` from your computer           \n---------------------------------------------")
    print(ex)


# In[2]:


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    ]
)

# Load dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

used_categories = range(len(classes))

## USE CODE BELOW IF YOUR COMPUTER IS TOO SLOW
reduce_dataset = True
if reduce_dataset:
    used_categories = (3, 5) # cats and dogs

    classes = [classes[i] for i in used_categories]
    new_train_data = []
    new_train_labels = []

    new_test_data = []
    new_test_labels = []
    for i, t in enumerate(used_categories):
        new_train_data.append(trainset.data[np.where(np.array(trainset.targets) == t)])
        new_train_labels += [i for _ in range(new_train_data[-1].shape[0])]

        new_test_data.append(testset.data[np.where(np.array(testset.targets) == t)])
        new_test_labels += [i for _ in range(new_test_data[-1].shape[0])]

    new_train_data = np.concatenate(new_train_data, 0)
    trainset.data = new_train_data
    trainset.targets = new_train_labels

    new_test_data = np.concatenate(new_test_data, 0)
    testset.data = new_test_data
    testset.targets = new_test_labels

# Batch size is set to 4 
batch_size = 64

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=True)
train_data_iter = iter(trainloader)
test_data_iter = iter(testloader)
print('used classes:', classes)


# In[3]:


print("# Training data")
print("Number of points:", len(trainset))
x, y = next(iter(trainloader))
print("Batch dimension [B x C x H x W]:", x.shape)
print("Number of distinct labels:", len(set(trainset.targets)))


print("\n# Test data")
print("Number of points:", len(testset))
x, y = next(iter(testloader))
print("Batch dimension [B x C x H x W]:", x.shape)
print("Number of distinct labels:", len(set(testset.targets)))


# In[4]:


# Run this cell multiple time to see more samples

def imshow(img):
    """ show an image """
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some random training images
images, labels = train_data_iter.next()

# show images
imshow(torchvision.utils.make_grid(images))
#print(images)

# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


# In[5]:


from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

from torch.nn import Linear, Conv2d, BatchNorm2d, MaxPool2d, Dropout2d
from torch.nn.functional import relu, elu, relu6, sigmoid, tanh, softmax


# ## Network Structure

# In[52]:


# Image dimensions
channels = x.shape[1]
height = x.shape[2]
width = x.shape[3]
num_features = (height * width) * channels

def compute_conv_dim(dim_size,kernel_size,padding_size,stride_size):
    return int((dim_size - kernel_size + 2 * padding_size) / stride_size + 1)

# First convolution layer
num_filters_conv1 = 16
kernel_size_conv1 = 5 # [height, width]
stride_conv1 = 1 # [stride_height, stride_width]
padding_conv1 = 2
input_dim = 32

# Second convolution layer
num_filters_conv2 = 5
kernel_size_conv2 = 5
stride_conv2 = 1
padding_conv2 = 1

# Convolutional layers
filters = [16,5]
kernels = [5,5]
padding = [2,1]
strides = [1,1]

# Encoder layer
h1e = 500

# Decoder layer
h1d = 1200
h2d = 1000

class AutoEncoder(nn.Module):
    def __init__(self, latent_features=2):
        super(AutoEncoder, self).__init__()
        # We typically employ an "hourglass" structure
        # meaning that the decoder should be an encoder
        # in reverse.

        # Encoder
        #self.encoder = nn.Sequential(
        #    nn.Conv2d(in_channels=channels, 
        #              out_channels=num_filters_conv1,
        #              kernel_size=kernel_size_conv1, 
        #              stride=stride_conv1,
        #             padding=padding_conv1),
        #    nn.ReLU(),
        #    nn.Linear(in_features=num_filters_conv1*conv_out_height*conv_out_width, out_features=h1e),
        #    nn.ReLU(),
        #    # bottleneck layer
        #    nn.Linear(in_features=h1e, out_features=latent_features)
        #)
        
        #self.conv_1 = Conv2d(in_channels=channels, 
        #              out_channels=num_filters_conv1,
        #              kernel_size=kernel_size_conv1, 
        #              stride=stride_conv1,
        #              padding=padding_conv1)

        self.conv_1 = Conv2d(in_channels=channels, 
                      out_channels=filters[0],
                      kernel_size=kernels[0], 
                      stride=strides[0],
                      padding=padding[0])
        
        # Output from first convolutional layer
        self.conv1_height = compute_conv_dim(height,kernels[0],padding[0],strides[0])
        self.conv1_width = compute_conv_dim(width,kernels[0],padding[0],strides[0])
        
        self.conv_2 = Conv2d(in_channels=filters[0],
                            out_channels=filters[1],
                            kernel_size=kernels[1],
                            stride=strides[1],
                            padding=padding[1])
        
        # Output from last convolutional layer
        self.conv_out_height = compute_conv_dim(self.conv1_height,kernels[-1],padding[-1],strides[-1])
        self.conv_out_width = compute_conv_dim(self.conv1_width,kernels[-1],padding[-1],strides[-1])
        
        
        # Linear layers
        self.l1_in_features = num_filters_conv2*self.conv_out_height*self.conv_out_width

        self.l_1 = Linear(in_features = self.l1_in_features,
                         out_features = h1e,
                         bias=True)
        
        self.l_out = Linear(in_features = h1e,
                           out_features = latent_features,
                           bias=False)

        # Decoder
        #self.decoder = nn.Sequential(
        #    nn.Linear(in_features=latent_features, out_features=h1d),
        #    nn.ReLU(),
        #    nn.Linear(in_features=h1d, out_features=h2d),
        #    nn.ReLU(),
        #    # output layer, projecting back to image size
        #    nn.Linear(in_features=h2d, out_features=num_features)
        #)

        self.l_1_de = Linear(in_features = latent_features,
                             out_features = h1d,
                             bias=True)

        self.conv_1_transpose = nn.ConvTranspose2d(in_channels=3, out_channels=3,
                            kernel_size=[13,13],
                            stride=[1,1],
                            padding=0, output_padding=0)



    def forward(self, x): 
        outputs = {}
        # we don't apply an activation to the bottleneck layer

        #z = self.encoder(x)
        print(np.shape(x))
        z = relu(self.conv_1(x))
        print(np.shape(z))
        z = relu(self.conv_2(z))
        print(np.shape(z))
        z = z.view(-1, self.l1_in_features)
        print(np.shape(z))
        z = relu(self.l_1(z))
        print(np.shape(z))
        z = relu(self.l_out(z))
        print(np.shape(z))

        x_hat = self.l_1_de(z)
        print(np.shape(x_hat))
        x_hat = x_hat.view(x_hat.size(0), channels, 20, 20)
        print(np.shape(x_hat))
        x_hat = self.conv_1_transpose(x_hat)
        print(np.shape(x_hat))

        x_hat = torch.sigmoid(x_hat)
        
        return {
            'z': z,
            'x_hat': x_hat
        }


# Choose the shape of the autoencoder
net = AutoEncoder(latent_features=300)

if cuda:
    net = net.cuda()

print(net)


# ## Loss function and optimizers

# In[53]:


import torch.optim as optim

# if you want L2 regularization, then add weight_decay to SGD
optimizer = optim.SGD(net.parameters(), lr=0.25)

# We will use pixel wise mean-squared error as our loss function
loss_function = nn.MSELoss()


# ## Test of one forward pass

# In[54]:


# test the forward pass
# expect output size of [32, num_features]
x, y = next(iter(trainloader))
imshow(torchvision.utils.make_grid(x))
#print(f"x.shape = {x.shape}")
#print(x)
#print(f"y.shape = {y.shape}")
#print(y)


# In[55]:


if cuda:
    x = x.cuda()

outputs = net(x)
print(f"x.shape = {x.shape}")
print(f"x_hat.shape = {outputs['x_hat'].shape}")
print(type(x))
x_hat = outputs['x_hat']
print(type(x_hat))
mm = x.view(-1, width*height*channels)
print(f"mm.shape = {mm.shape}")


# In[37]:


from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

num_epochs = 5

train_loss = []
valid_loss = []
counter = 0
for epoch in range(num_epochs):
    batch_loss = []
    net.train()
    
    # Go through each batch in the training dataset using the loader
    # Note that y is not necessarily known as it is here
    for x, y in trainloader:
        
        if cuda:
            x = x.cuda()

        #if counter == 156:
        #    print("STOP")
        outputs = net(x)
        x_hat = outputs['x_hat']

        # note, target is the original tensor, as we're working with auto-encoders
        #loss = loss_function(x_hat, x.view(-1, width*height*channels)) # TC: Changes this
        loss = loss_function(x_hat, x)  #

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_loss.append(loss.item())
        #counter = counter + 1
        #print("Counter", counter)

    train_loss.append(np.mean(batch_loss))

    # Evaluate, do not propagate gradients
    with torch.no_grad():
        net.eval()
        
        # Just load a single batch from the test loader
        x, y = next(iter(testloader))
        
        if cuda:
            x = x.cuda()
        
        outputs = net(x)

        # We save the latent variable and reconstruction for later use
        # we will need them on the CPU to plot
        x_hat = outputs['x_hat']
        z = outputs['z'].cpu().numpy()

        #loss = loss_function(x_hat, x.view(batch_size, width*height*channels))
        loss = loss_function(x_hat, x)

        valid_loss.append(loss.item())
    
    if epoch == 0:
        continue

    print("Epoch", epoch, "/", num_epochs)
# live plotting of the trainig curves and representation


# In[38]:


#plot_autoencoder_stats(x=x.view(batch_size, width*height*channels),
#                        x_hat=x_hat,
#                        z=z,
#                        y=y,
#                        train_loss=train_loss,
#                        valid_loss=valid_loss,
#                        epoch=epoch,
#                        classes=classes,
#                        dimensionality_reduction_op=None)


# In[13]:


#imshow(torchvision.utils.make_grid(x.cpu()))
x_hat = x_hat.cpu()
#x_hat_r = x_hat[:, 0:1024].view(batch_size, width, height)
#x_hat_g = x_hat[:, 1024:2048].view(batch_size, width, height)
#x_hat_b = x_hat[:, 2048:3072].view(batch_size, width, height)
#x_hat_rgb = torch.stack((x_hat_r, x_hat_g, x_hat_b), dim=1)

def unnormalize_image(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    return np.transpose(npimg, (1, 2, 0))

f, axarr = plt.subplots(1, 2, figsize=(20, 20))

ax = axarr[0]
x_grid = torchvision.utils.make_grid(x.cpu())
ax.imshow(unnormalize_image(x_grid))

ax = axarr[1]
x_hat_grid = torchvision.utils.make_grid(x_hat)
#x_hat_grid = torchvision.utils.make_grid(x_hat_rgb)
ax.imshow(unnormalize_image(x_hat_grid))

#imshow(torchvision.utils.make_grid(x.cpu()))


plt.show()
#g = torchvision.utils.make_grid(x)


# In[ ]:




