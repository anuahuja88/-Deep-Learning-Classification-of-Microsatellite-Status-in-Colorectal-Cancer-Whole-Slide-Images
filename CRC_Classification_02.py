import torch
import torchvision.transforms as T
from torchvision.models import resnet18
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torch.optim import SGD, Adam
from torchmetrics import Accuracy
import torch.nn as nn


# Set seed.
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# Learning and training parameters.
epochs = 100
batch_size = 32
learning_rate = 0.01
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Model
model = resnet18()

for params in model.parameters():
    params.requires_grad = True

model.fc = nn.Linear(in_features=512, out_features=1)

# Optimizer.
optimizer = Adam(model.parameters(), lr=learning_rate)

# Loss function.
loss_fn = nn.BCEWithLogitsLoss()  #pos_weight=torch.ones([batch_size]))

transform = T.Compose([ 
    T.Resize(256),
    T.RandomCrop(224),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ToTensor()   #Required!
]) 

dataset = ImageFolder('train_crc', transform=transform)
size = len(dataset)     #408  (204 mss + 204 msi)
train_data_len = int(size*0.8)
valid_data_len = int(size - train_data_len)
train_data, val_data = random_split(dataset, [train_data_len, valid_data_len])
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader)          # 11 batches (326/32) each of size 32

    model.train()
    running_loss = 0
    avg_loss = 0
    for i, data in enumerate(dataloader):
        inputs, labels = data
        outputs = model(inputs)    #torch.Size[32, 1]
        labels = labels.float()    #torch.Size[32]
        preds = outputs[:, 0]      #torch.Size[32]
        
        # Zero your gradients for every batch!
        optimizer.zero_grad()
        # Compute the loss and its gradients
        loss = loss_fn(preds, labels)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()
        #print('Loss for batch ', i, ' = ', loss.item())

    avg_loss = running_loss/size
    print("Average loss in this epoch =", avg_loss)


def test_loop(dataloader, model):

    model.eval()
    num_batches = len(dataloader)
    accuracy = 0
    total_acc = 0

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs, labels = data
            outputs = model(inputs)
            preds = outputs[:, 0]      #torch.Size[32]
            preds = torch.sigmoid(preds)
            accuracy = (preds.round() == labels).float().mean()  #Mean accuracy value per batch
            print('Average accuracy for batch ', i, ' = ', accuracy.item())
            total_acc += accuracy

    total_acc /= num_batches
    print('Total Average Accuracy = ', total_acc.item())



# Start the training.
for epoch in range(epochs):
    print(f"[INFO]: Epoch {epoch+1} of {epochs}")
    train_loop(train_loader, model, loss_fn, optimizer)

test_loop(valid_loader, model)