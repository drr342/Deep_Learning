
# coding: utf-8

# ## 0. Setup

# In[1]:


# Import dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# In[6]:


# Set up your device 
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")


# In[7]:


# Set up random seed to 1008. Do not change the random seed.
seed = 1008
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed_all(seed)


# ## 1. Data: MNIST
# #### Load the MNIST training and test dataset using $\texttt{torch.utils.data.DataLoader}$ and $\texttt{torchvision.datasets}$. 
# 
# Hint: You might find Alf's notebook useful: https://github.com/Atcold/pytorch-Deep-Learning-Minicourse/blob/master/06-convnet.ipynb

# ### 1.1. Load Training Set [4 pts]

# In[77]:


# Load the MNIST training set with batch size 128, apply data shuffling and normalization
mnist_train = datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([transforms.ToTensor()]))
# print(vars(mnist_train))
mean = mnist_train.train_data.type(torch.FloatTensor).mean() / 255
std = mnist_train.train_data.type(torch.FloatTensor).std() / 255
mnist_train.transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((mean.item(),), (std.item(),)) 
                        ])
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=128, shuffle=True)


# ### 1.1. Load Test Set [4 pts]

# In[79]:


# Load the MNIST test set with batch size 128, apply data shuffling and normalization
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((mean.item(),), (std.item(),))
                   ])),
    batch_size=128, shuffle=True)


# ## 2. Models
# #### You are going to define two convolutional neural networks which are trained to classify MNIST digits

# ### 2.1. CNN without Batch Norm [15 pts]

# In[143]:


# print(train_loader.__iter__().next()[0].shape)
# x = torch.randn(128, 1, 28, 28)
# conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1)
# x = F.relu(conv1(x))
# print(x.shape)
# x = F.max_pool2d(x, kernel_size=2, stride=2)
# print(x.shape)
# conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1)
# x = F.relu(conv2(x))
# print(x.shape)
# x = F.max_pool2d(x, kernel_size=2, stride=2)
# print(x.shape)
# x = x.view(-1, 50*4*4)
# print(x.shape)
# x = F.softmax(x, dim=1)
# print(x.shape)
# print(x.sum(1))

# torch.Size([128, 20, 24, 24])
# torch.Size([128, 20, 12, 12])
# torch.Size([128, 50, 8, 8])
# torch.Size([128, 50, 4, 4])
# torch.Size([128, 800])
# torch.Size([128, 800])


# In[115]:


# Fill in the values below that make this network valid for MNIST data

conv1_in_ch = 1
conv2_in_ch = 20
fc1_in_features = 800
fc2_in_features = 500
n_classes = 10


# In[140]:


class NetWithoutBatchNorm(nn.Module):
    def __init__(self):
        super(NetWithoutBatchNorm, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=conv1_in_ch, out_channels=20, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=conv2_in_ch, out_channels=50, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(in_features=fc1_in_features, out_features=500)
        self.fc2 = nn.Linear(in_features=fc2_in_features, out_features=n_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, fc1_in_features) # reshaping
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # Return the log_softmax of x.
        return F.log_softmax(x, dim=1)


# ### 2.2. CNN with Batch Norm [15 pts]

# In[149]:


# Fill in the values below that make this network valid for MNIST data

conv1_bn_size = 20
conv2_bn_size = 50
fc1_bn_size = 500


# In[141]:


# Define the CNN with architecture explained in Part 2.2
class NetWithBatchNorm(nn.Module):
    def __init__(self):
        super(NetWithBatchNorm, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=conv1_in_ch, out_channels=20, kernel_size=5, stride=1)
        self.conv1_bn = nn.BatchNorm2d(conv1_bn_size)
        self.conv2 = nn.Conv2d(in_channels=conv2_in_ch, out_channels=50, kernel_size=5, stride=1)
        self.conv2_bn = nn.BatchNorm2d(conv2_bn_size)
        self.fc1 = nn.Linear(in_features=fc1_in_features, out_features=500)
        self.fc1_bn = nn.BatchNorm1d(fc1_bn_size)
        self.fc2 = nn.Linear(in_features=fc2_in_features, out_features=n_classes)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, fc1_in_features)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.fc2(x)
        # Return the log_softmax of x.
        return F.log_softmax(x, dim=1)


# ## 3. Training & Evaluation

# ### 3.1. Define training method [15 pts]

# In[145]:


def train(model, device, train_loader, optimizer, epoch, log_interval = 100):
    # Set model to training mode
    model.train()
    # Loop through data points
    for batch_idx, (data, target) in enumerate(train_loader):
        # pass # remove once implemented
    
        # Send data and target to device
        data, target = data.to(device), target.to(device)
        
        # Zero out the optimizer
        optimizer.zero_grad()
        
        # Pass data through model
        output = model(data)
        
        # Compute the negative log likelihood loss
        loss = F.nll_loss(output, target)
        
        # Backpropagate loss
        loss.backward()
        
        # Make a step with the optimizer
        optimizer.step()
        
        # Print loss (uncomment lines below once implemented)
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


# ### 3.2. Define test method [15 pts]

# In[146]:


# Define test method
def test(model, device, test_loader):
    # Set model to evaluation mode
    model.eval()
    # Variable for the total loss 
    test_loss = 0
    # Counter for the correct predictions
    num_correct = 0
    
    with torch.no_grad():
        # Loop through data points
        for data, target in test_loader:
            pass # remove once implemented
        
            # Send data to device
            data, target = data.to(device), target.to(device)
            
            # Pass data through model
            output = model(data) 
            
            # Compute the negative log likelihood loss with reduction='sum' and add to total test_loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            
            # Get predictions from the model for each data point
            pred = output.data.max(1, keepdim=True)[1] 
            
            # Add number of correct predictions to total num_correct 
            num_correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
    
    # Compute the average test_loss
    avg_test_loss = test_loss / len(test_loader.dataset)
    
    # Print loss (uncomment lines below once implemented)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_test_loss, num_correct, len(test_loader.dataset),
        100. * num_correct / len(test_loader.dataset)))


# ### 3.3 Train NetWithoutBatchNorm() [5 pts]

# In[147]:


# Deifne model and sent to device
model = NetWithoutBatchNorm()
model.to(device)

# Optimizer: SGD with learning rate of 1e-2 and momentum of 0.5
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.5)

# Training loop with 10 epochs
for epoch in range(1, 10 + 1):
    # pass # remove once implemented

    # Train model
    train(model, device, train_loader, optimizer, epoch)
    
    # Test model
    test(model, device, test_loader)


# ### 3.4 Train NetWithBatchNorm() [5 pts]

# In[150]:


# Deifne model and sent to device
model = NetWithBatchNorm()
model.to(device)

# Optimizer: SGD with learning rate of 1e-2 and momentum of 0.5
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.5)

# Training loop with 10 epochs
for epoch in range(1, 10 + 1):
    # pass # remove once implemented

    # Train model
    train(model, device, train_loader, optimizer, epoch)
    
    # Test model
    test(model, device, test_loader)


# ## 4. Empirically, which of the models achieves higher accuracy faster? [2 pts]

# Answer: The CNN with batch normalization achieves higher accuracy faster. After only one epoch is already at the same level of accuracy (97.7%) that the CNN without batch normalization achieves at epoch 3. Furthermore, after 7 epochs the CNN with batch normalization seems to have reached it maximum accuracy (99.2%).
