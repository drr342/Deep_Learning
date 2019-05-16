#!/usr/bin/env python
# coding: utf-8

# Import dependencies
import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from datetime import datetime

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def deconv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, output_padding=0):
    """3x3 deconvolution with padding"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation, output_padding=output_padding)

def deconv1x1(in_planes, out_planes, stride=1, output_padding=0):
    """1x1 deconvolution"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=stride, 
                              bias=False, output_padding=output_padding)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, mode, inplanes, planes, stride=1, resample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, output_padding=0):
        super(BasicBlock, self).__init__()
        if mode not in('conv', 'deconv'):
            raise ValueError('BasickBlock mode can only be conv or deconv')
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if mode == 'conv':
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.conv2 = conv3x3(planes, planes)
            self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        else:
            self.conv1 = deconv3x3(inplanes, planes, stride, output_padding=output_padding)
            self.conv2 = deconv3x3(planes, planes)
            self.relu = nn.ReLU(inplace=True)
        self.bn1 = norm_layer(planes)
        self.bn2 = norm_layer(planes)
        self.resample = resample
        self.stride = stride

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.resample is not None:
            identity = self.resample(x)
        
        out += identity
        out = self.relu(out)
        return out


class ResNetVAE(nn.Module):

    def __init__(self, layers, groups=1, width_per_group=64):
                 
        super(ResNetVAE, self).__init__()
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer('conv', 64, layers[0])
        self.layer2 = self._make_layer('conv', 128, layers[1], stride=2)               
        self.layer3 = self._make_layer('conv', 256, layers[2], stride=2)
        self.layer4 = self._make_layer('conv', 512, layers[3], stride=2)
        
        self.fc1 = nn.Linear(18432, 512)
        self.fc2 = nn.Linear(18432, 512)
        self.fc3 = nn.Linear(512, 18432)
        
        self.decon4 = self._make_layer('deconv', 512, layers[3], stride=2, output_padding=1)
        self.decon3 = self._make_layer('deconv', 256, layers[2], stride=2, output_padding=1)
        self.decon2 = self._make_layer('deconv', 128, layers[1], stride=2, output_padding=1)
        self.decon1 = self._make_layer('deconv', 64, layers[0], output_padding=0)
        
        self.decon0 = nn.ConvTranspose2d(self.inplanes, 3, kernel_size=7, stride=2, padding=3,
                               bias=False, output_padding=1)
        self.bn0 = self._norm_layer(3)
        self.sigmoid = nn.Sigmoid()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, mode, planes, blocks, stride=1, output_padding=0):
        norm_layer = self._norm_layer
        resample = None
        previous_dilation = self.dilation
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            if mode == 'conv':
                resample = nn.Sequential(
                    conv1x1(self.inplanes, planes * BasicBlock.expansion, stride),
                    norm_layer(planes * BasicBlock.expansion),
                )
            else:
                resample = nn.Sequential(
                    deconv1x1(self.inplanes, planes * BasicBlock.expansion, stride, output_padding=output_padding),
                    norm_layer(planes * BasicBlock.expansion),
                )

        layers = []
        layers.append(BasicBlock(mode, self.inplanes, planes, stride, resample, self.groups,
                            self.base_width, previous_dilation, norm_layer, output_padding=output_padding))
        self.inplanes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(mode, self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
    
    def _encode(self, x):
        # [N, 3, 96, 96]
        x = self.conv1(x)
        # [N, 64, 48, 48]
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        # [N, 64, 48, 48]
        x = self.layer2(x)
        # [N, 128, 24, 24]
        x = self.layer3(x)
        # [N, 256, 12, 12]
        x = self.layer4(x)
        # [N, 512, 6, 6]
        return x.view(x.size(0), -1)
    
    def _decode(self, x):
        # [N, 18432]
        x = x.view(x.size(0), 512, 6, 6)
        x = self.decon4(x)
        # [N, 512, 12, 12]
        x = self.decon3(x)
        # [N, 256, 24, 24]
        x = self.decon2(x)
        # [N, 128, 48, 48]
        x = self.decon1(x)
        # [N, 64, 48, 48]
        
        x = self.decon0(x)
        # [N, 3, 96, 96]
        x = self.bn0(x)
        # x = self.sigmoid(x)
        return x

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
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
        
def _transform(resize=False, size=224):
    if not resize:
        return transforms.ToTensor()
    return transforms.Compose([
        transforms.Resize(size, 1),
        transforms.ToTensor()
        ])

def load_data(data_dir, batch_size, size=224, shuffle=True, **kwargs):
    """ Method returning a data loader for labeled data """
    # TODO (optional): add data transformations if needed
#     transform = transforms.ToTensor()
    data = datasets.ImageFolder(f'{data_dir}', transform=_transform())
    data_loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        **kwargs
    )
    return data_loader


batch_size = 128
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
data_loader = load_data('/scratch/drr342/dl/project/data/ssl_data_96/unsupervised', batch_size, True, kwargs)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, batch_idx, epoch):
    div = 3 * 96 * 96 * x.size(0)
    BCE = F.binary_cross_entropy_with_logits(recon_x, x, reduction='sum') / div
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum((1 + logvar - logvar.exp()) / div - (mu / (div ** 0.5)).pow(2))
    if epoch > 1:
        beta = 1
    else:
        beta = batch_idx / len(data_loader)
    return BCE + beta * KLD

model = ResNetVAE([2, 2, 2, 2])
if torch.cuda.device_count() > 1:
    # print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

train_loss = 0
log_interval = 100
fileName = f'ResNetVAE_progress_{datetime.now().strftime("%Y%m%d_%H%M")}.txt'

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(data_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar, batch_idx, epoch)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % log_interval == 0:
            f= open(fileName, "a+")
            f.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\n'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader),
                loss.item() / len(data)))
            f.close()

    f= open(fileName,"a+")
    f.write('====> Epoch: {} Average loss: {:.4f}\n'.format(
          epoch, train_loss / len(data_loader.dataset)))
    f.write("Saving model and checkpoint of progress...\n\n")
    f.close()


epochs = 5
f= open(fileName, "a+")
f.write("ResNet-based VAE: Starting Training...\n\n")
f.close()
for epoch in range(1, epochs + 1):
    train(epoch)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss
        }, f'/scratch/drr342/dl/project/models/ResNetVAE_{datetime.now().strftime("%Y%m%d_%H%M")}.pth')

