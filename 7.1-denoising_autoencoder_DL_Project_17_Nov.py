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
    used_categories = (3, 5, 9, 0) # cats and dogs

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

print(compute_conv_dim(32, 5, 2, 1))

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
filters = [16,32,64]
kernels = [5,6,6]
padding = [2,2,2]
strides = [1,2,2]

# Encoder layer
h1e = 64*8*8

# Decoder layer
h1d = 64*8*8
h2d = 1000

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        #self.pool0 = nn.MaxPool2d(3, 2, padding=1)  # 256 -> 128
        self.enc_norm0 = nn.BatchNorm2d(32)
        self.enc_conv1 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        #self.pool1 = nn.MaxPool2d(3, 2, padding=1)  # 128 -> 64
        self.enc_norm1 = nn.BatchNorm2d(64)
        self.enc_conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        #self.pool2 = nn.MaxPool2d(3, 2, padding=1)  # 64 -> 32
        self.enc_norm2 = nn.BatchNorm2d(128)
        self.enc_conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        #self.pool3 = nn.MaxPool2d(3, 2, padding=1)  # 32 -> 16
        self.enc_norm3 = nn.BatchNorm2d(256)

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.norm_bottleneck = nn.BatchNorm2d(256)

        # decoder (upsampling)
        #self.upsample0 = nn.Upsample(32)  # 16 -> 32
        #self.dec_conv0 = nn.Conv2d(128, 64, 3, padding=1)
        #self.upsample1 = nn.Upsample(64)  # 32 -> 64
        #self.dec_conv1 = nn.Conv2d(128, 64, 3, padding=1)
        #self.upsample2 = nn.Upsample(128)  # 64 -> 128
        #self.dec_conv2 = nn.Conv2d(128, 64, 3, padding=1)
        #self.upsample3 = nn.Upsample(256)  # 128 -> 256
        #self.dec_conv3 = nn.Conv2d(128, 1, 3, padding=1)


        self.dec_conv0 = nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1)
        self.dec_norm0 = nn.BatchNorm2d(256)
        self.dec_conv1 = nn.ConvTranspose2d(256+128, 128, 3, stride=2, padding=1, output_padding=1)
        self.dec_norm1 = nn.BatchNorm2d(128)
        self.dec_conv2 = nn.ConvTranspose2d(128+64, 64, 3, stride=2, padding=1, output_padding=1)
        self.dec_norm2 = nn.BatchNorm2d(64)
        self.dec_conv3 = nn.ConvTranspose2d(64+32, 32, 3, stride=2, padding=1, output_padding=1)
        self.dec_norm3 = nn.BatchNorm2d(32)

        self.final_deconv = nn.ConvTranspose2d(32, 3, 3, stride=1, padding=1, output_padding=0)


    def forward(self, x):
        # encoder
        e0 = F.relu(self.enc_norm0(self.enc_conv0(x)))
        e1 = F.relu(self.enc_norm1(self.enc_conv1(e0)))
        e2 = F.relu(self.enc_norm2(self.enc_conv2(e1)))
        e3 = F.relu(self.enc_norm3(self.enc_conv3(e2)))

        # bottleneck
        b = F.relu(self.norm_bottleneck(self.bottleneck_conv(e3)))
        z = b

        # decoder
        #skip0 = torch.cat([self.upsample0(b), F.relu(self.enc_conv3(e2))], 1)
        #d0 = F.relu(self.dec_conv0(skip0))
        #skip1 = torch.cat([self.upsample1(d0), F.relu(self.enc_conv2(e1))], 1)
        #d1 = F.relu(self.dec_conv1(skip1))
        #skip2 = torch.cat([self.upsample2(d1), F.relu(self.enc_conv1(e0))], 1)
        #d2 = F.relu(self.dec_conv2(skip2))
        #skip3 = torch.cat([self.upsample3(d2), F.relu(self.enc_conv0(x))], 1)
        #d3 = self.dec_conv3(skip3)  # no activation

        # decoder
        d0 = F.relu(self.dec_norm0(self.dec_conv0(b)))
        skip0 = torch.cat([d0, e2], 1)
        d1 = F.relu(self.dec_norm1(self.dec_conv1(skip0)))
        skip1 = torch.cat([d1, e1], 1)
        d2 = F.relu(self.dec_norm2(self.dec_conv2(skip1)))
        skip2 = torch.cat([d2, e0], 1)
        d3 = self.dec_conv3(skip2)  # no activation

        x_hat = self.final_deconv(d3)

        return {
            'z': z,
            'x_hat': x_hat
        }


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

        # Output from second convolutional layer
        self.conv2_height = compute_conv_dim(self.conv1_height, kernels[1], padding[1], strides[1])
        self.conv2_width = compute_conv_dim(self.conv1_width, kernels[1], padding[1], strides[1])

        self.conv_3 = Conv2d(in_channels=filters[1],
                             out_channels=filters[2],
                             kernel_size=kernels[2],
                             stride=strides[2],
                             padding=padding[2])

        # Output from last convolutional layer
        self.conv_out_height = compute_conv_dim(self.conv2_height,kernels[-1],padding[-1],strides[-1])
        self.conv_out_width = compute_conv_dim(self.conv2_width,kernels[-1],padding[-1],strides[-1])
        
        
        # Linear layers
        self.l1_in_features = filters[-1]*self.conv_out_height*self.conv_out_width

        #self.l_1 = Linear(in_features = self.l1_in_features,
        #                 out_features = h1e,
        #                 bias=True)
        
        self.l_out = Linear(in_features = self.l1_in_features,
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

        self.conv_1_transpose = nn.ConvTranspose2d(in_channels=64, out_channels=32,
                                kernel_size=[6,6],
                                stride=[2,2],
                                padding=2, output_padding=0)

        self.conv_2_transpose = nn.ConvTranspose2d(in_channels=32, out_channels=16,
                               kernel_size=[6, 6],
                               stride=[2, 2],
                               padding=2, output_padding=0)

        self.conv_3_transpose = nn.ConvTranspose2d(in_channels=16, out_channels=3,
                               kernel_size=[5, 5],
                               stride=[1, 1],
                               padding=2, output_padding=0)




    def forward(self, x): 
        outputs = {}
        # we don't apply an activation to the bottleneck layer

        #z = self.encoder(x)
        #print(np.shape(x))
        z = relu(self.conv_1(x))
        #print(np.shape(z))
        z = relu(self.conv_2(z))
        #print(np.shape(z))
        z = relu(self.conv_3(z))
        #print(np.shape(z))
        z = z.view(-1, self.l1_in_features)
        #print(np.shape(z))

        #z = relu(self.l_1(z))
        #print(np.shape(z))

        z = relu(self.l_out(z))
        #print(np.shape(z))

        x_hat = relu(self.l_1_de(z))
        #print(np.shape(x_hat))
        x_hat = x_hat.view(x_hat.size(0), 64, 8, 8)
        #print(np.shape(x_hat))
        x_hat = relu(self.conv_1_transpose(x_hat))
        #print(np.shape(x_hat))
        x_hat = relu(self.conv_2_transpose(x_hat))
        #print(np.shape(x_hat))
        x_hat = self.conv_3_transpose(x_hat)

        x_hat = torch.sigmoid(x_hat)
        
        return {
            'z': z,
            'x_hat': x_hat
        }


# Choose the shape of the autoencoder
#net = AutoEncoder(latent_features=75)
net = UNet()

if cuda:
    net = net.cuda()

print(net)


# ## Loss function and optimizers

# In[53]:


import torch.optim as optim

# if you want L2 regularization, then add weight_decay to SGD
#optimizer = optim.SGD(net.parameters(), lr=0.25)
optimizer = optim.Adam(net.parameters(), lr=0.01)

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

num_epochs = 3

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
        print(loss)

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




