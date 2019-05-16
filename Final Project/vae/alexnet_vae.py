#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import dependencies
import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms

from datetime import datetime


# In[2]:


cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")


# In[3]:


class AlexNetVAE(nn.Module):
    def __init__(self):
        super(AlexNetVAE, self).__init__()
        #Encoding layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.mp1   = nn.MaxPool2d(kernel_size=3, stride=2,return_indices=True)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.mp2   = nn.MaxPool2d(kernel_size=3, stride=2,return_indices=True)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # self.mp5   = nn.MaxPool2d(kernel_size=3, stride=2,return_indices=True)

        #Bottleneck layers
        self.fc1 = nn.Linear(6400, 64)
        self.fc2 = nn.Linear(6400, 64)
        self.fc3 = nn.Linear(64, 6400)
        
        #Decoding layers
        # self.mu5     = nn.MaxUnpool2d(kernel_size=3, stride=2)
        self.deconv5 = nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1)
        self.deconv4 = nn.ConvTranspose2d(256, 384, kernel_size=3, padding=1)
        self.deconv3 = nn.ConvTranspose2d(384, 192, kernel_size=3, padding=1)
        self.mu2     = nn.MaxUnpool2d(kernel_size=3, stride=2)
        self.deconv2 = nn.ConvTranspose2d(192, 64, kernel_size=5, padding=2)
        self.mu1     = nn.MaxUnpool2d(kernel_size=3, stride=2)
        self.deconv1 = nn.ConvTranspose2d(64, 3, kernel_size=11, stride=4, padding=2, output_padding=1)
        
    def _encode(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x, self.mp1_index = self.mp1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x, self.mp2_index = self.mp2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        # x, self.mp5_index = self.mp5(x)
        return x.view(x.size(0), -1)
    
    def _decode(self, x):
        x = x.view(x.size(0), 256, 5, 5)
        # x = self.mu5(x, self.mp5_index, output_size=torch.Size([x.size(0), 256, 13, 13]))
        x = self.deconv5(x)
        x = F.relu(x)
        x = self.deconv4(x)
        x = F.relu(x)
        x = self.deconv3(x)
        x = F.relu(x)
        x = self.mu2(x, self.mp2_index, output_size=torch.Size([x.size(0), 192, 11, 11]))
        x = self.deconv2(x)
        x = F.relu(x)
        x = self.mu1(x, self.mp1_index, output_size=torch.Size([x.size(0), 64, 23, 23]))
        x = self.deconv1(x)
        return torch.sigmoid(x)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
        
    def encode(self, x):
        h = self._encode(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self._decode(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar


# In[4]:


def load_data(data_dir, batch_size, shuffle=True, **kwargs):
    """ Method returning a data loader for labeled data """
    # TODO (optional): add data transformations if needed
    transform = transforms.Compose([
        # transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
        ]
    )
    data = datasets.ImageFolder(data_dir, transform=transform)
    data_loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        **kwargs
    )
    return data_loader


# In[5]:


batch_size = 128
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = load_data('/scratch/drr342/dl/project/data/ssl_data_96/unsupervised', batch_size)


# In[6]:


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, batch_idx):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    beta = (batch_idx + 1) / len(train_loader)
    return BCE + beta * KLD


# In[9]:


log_interval=100
model = AlexNetVAE()
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

# pretrained_model = torch.load(f='/scratch/drr342/dl/project/models/AlexNetVAE_20190503_2159.pth', map_location="cuda" if cuda else "cpu")
# model.load_state_dict(pretrained_model, strict=True)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar, batch_idx)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


# In[10]:


epochs=10
for epoch in range(1, epochs + 1):
    train(epoch)
    torch.save(model.state_dict(), f'/scratch/drr342/dl/project/models/AlexNetVAE_{datetime.now().strftime("%Y%m%d_%H%M")}.pth')


# In[ ]:




