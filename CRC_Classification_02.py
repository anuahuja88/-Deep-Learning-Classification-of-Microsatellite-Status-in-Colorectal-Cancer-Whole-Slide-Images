# import torch
# import torchvision.transforms as T
# from torchvision.models import resnet34
# from torchvision.datasets import ImageFolder
# from torch.utils.data import DataLoader, random_split
# from torch.optim import Adam
# from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC
# import torch.nn as nn
# import torch.nn.functional as F

# # Set seed.
# seed = 42
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = True

# # Learning and training parameters.
# epochs = 20
# batch_size = 32
# learning_rate = 0.01
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# # Model
# model = resnet34()
# model.fc = nn.Linear(in_features=512, out_features=1)
# model.to(device)

# # Optimizer.
# optimizer = Adam(model.parameters(), lr=learning_rate)

# # Loss function.
# loss_fn = nn.BCEWithLogitsLoss()

# # Transformations
# transform = T.Compose([
#     T.Resize(256),
#     T.RandomCrop(224),
#     T.RandomHorizontalFlip(),
#     T.RandomVerticalFlip(),
#     T.ToTensor()
# ])

# # Dataset and DataLoader
# dataset = ImageFolder('train_crc', transform=transform)
# size = len(dataset)
# train_data_len = int(size * 0.8)
# valid_data_len = size - train_data_len
# train_data, val_data = random_split(dataset, [train_data_len, valid_data_len])
# train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
# valid_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

# # Metrics for binary classification
# accuracy_metric = Accuracy(task="binary").to(device)
# precision_metric = Precision(task="binary").to(device)
# recall_metric = Recall(task="binary").to(device)
# f1_metric = F1Score(task="binary").to(device)
# roc_metric = AUROC(task="binary").to(device)

# def train_loop(dataloader, model, loss_fn, optimizer):
#     model.train()
#     running_loss = 0

#     for inputs, labels in dataloader:
#         inputs, labels = inputs.to(device), labels.to(device).float()

#         # Forward pass
#         outputs = model(inputs).squeeze()
#         loss = loss_fn(outputs, labels)

#         # Backward pass and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()

#     avg_loss = running_loss / len(dataloader)
#     print(f"Training Loss: {avg_loss:.4f}")
#     return avg_loss

# def test_loop(dataloader, model):
#     model.eval()
#     running_loss = 0
#     accuracy_metric.reset()
#     precision_metric.reset()
#     recall_metric.reset()
#     f1_metric.reset()
#     roc_metric.reset()

#     with torch.no_grad():
#         for inputs, labels in dataloader:
#             inputs, labels = inputs.to(device), labels.to(device).float()

#             outputs = model(inputs).squeeze()
#             preds = torch.sigmoid(outputs)

#             # Calculate loss
#             loss = loss_fn(outputs, labels)
#             running_loss += loss.item()

#             # Update metrics
#             accuracy_metric(preds, labels.int())
#             precision_metric(preds, labels.int())
#             recall_metric(preds, labels.int())
#             f1_metric(preds, labels.int())
#             roc_metric(preds, labels.int())

#     avg_loss = running_loss / len(dataloader)
#     accuracy = accuracy_metric.compute().item()
#     precision = precision_metric.compute().item()
#     recall = recall_metric.compute().item()
#     f1 = f1_metric.compute().item()
#     roc = roc_metric.compute().item()

#     print(f"Validation Loss: {avg_loss:.4f}")
#     print(f"Accuracy: {accuracy:.4f}")
#     print(f"Precision: {precision:.4f}")
#     print(f"Recall: {recall:.4f}")
#     print(f"F1 Score: {f1:.4f}")
#     print(f"ROC AUC: {roc:.4f}")

#     return avg_loss, accuracy, precision, recall, f1, roc

# # Start the training.
# for epoch in range(epochs):
#     print(f"[INFO]: Epoch {epoch+1} of {epochs}")
#     train_loop(train_loader, model, loss_fn, optimizer)

# # Evaluate on validation set
# test_loop(valid_loader, model)

# # Save the model
# torch.save(model.state_dict(), 'resnet34.pth')
import torch
import torchvision.transforms as T
from torchvision.models import resnet34
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
epochs = 20
batch_size = 32
learning_rate = 0.01
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Model
model = resnet34()

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


# Save the model
torch.save(model.state_dict(), 'resnet34.pth')