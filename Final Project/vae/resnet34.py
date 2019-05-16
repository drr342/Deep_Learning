# Import dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet

# Set up your device 
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

# Data loader
def load_data(data_dir, batch_size, split):
    """ Method returning a data loader for labeled data """
    # TODO (optional): add data transformations if needed
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
        ]
    )
    data = datasets.ImageFolder(f'{data_dir}/supervised/{split}', transform=transform)
    data_loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    return data_loader

# Deifne model and sent to device
model = resnet.resnet34(pretrained=False)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

# model = nn.DataParallel(model)
pretrained_model = torch.load(f='/scratch/drr342/dl/project/models/resnet34.pth', map_location="cuda" if cuda else "cpu")
# with torch.no_grad():
model.load_state_dict(pretrained_model, strict=True)

model.to(device)
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.5)
train_loader = load_data('/scratch/drr342/dl/project/data/ssl_data_96', 64, 'train')

# Training loop
model.train()
epochs = 50
log_interval = 100
for epoch in range(1, epochs + 1):
    for batch_idx, (data, target) in enumerate(train_loader):    
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        # Print loss (uncomment lines below once implemented)
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    print("Saving model...")
    torch.save(model.state_dict(), '/scratch/drr342/dl/project/models/resnet34.pth')

