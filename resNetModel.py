import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Define transformations
train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_transforms_horizontal = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_transforms_vertical = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.56, saturation=0.2, hue=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load datasets
train_dataset = torchvision.datasets.ImageFolder(root='./Data/train', transform=train_transforms)
horizontal_dataset = torchvision.datasets.ImageFolder(root='./Data/train', transform=train_transforms_horizontal)
vertical_dataset = torchvision.datasets.ImageFolder(root='./Data/train', transform=train_transforms_vertical)

print("Normal: ", len(train_dataset))
print("Horizontal: ", len(horizontal_dataset))
print("Vertical: ", len(vertical_dataset))
val_dataset = torchvision.datasets.ImageFolder(root='./Data/val', transform=val_transforms)

train_dataset = ConcatDataset([train_dataset, horizontal_dataset, vertical_dataset])
print(len(train_dataset))
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

class CancerClassifier(pl.LightningModule):
    def __init__(self):
        super(CancerClassifier, self).__init__()
        self.model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        # Modify the final fully connected layer for binary classification
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 1)
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images).squeeze()
        loss = nn.BCEWithLogitsLoss()(outputs, labels.float())
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images).squeeze()
        loss = nn.BCEWithLogitsLoss()(outputs, labels.float())
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        return [optimizer], [scheduler]

def test_model(model, test_data_path):
    # Define test transforms
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load test dataset
    test_dataset = torchvision.datasets.ImageFolder(root=test_data_path, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Put the model in evaluation mode
    model.eval()
    
    # Perform evaluation on the test set
    test_accuracy = torchmetrics.Accuracy(task="binary").to('cuda:0')
    test_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    all_preds = []
    all_labels = []
    torch.cuda.empty_cache()
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch
            images = images.to("cuda:0")  # Move images to the GPU
            labels = labels.to("cuda:0")  # Move labels to the GPU
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels.float())
            test_loss += loss.item()
            preds = torch.sigmoid(outputs) > 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            test_accuracy.update(preds.float(), labels.float())
    
    test_loss /= len(test_loader)
    print(f'Test Loss: {test_loss}')
    print(f'Test Accuracy: {test_accuracy.compute()}')
    
    return all_preds, all_labels


def visualize_predictions(model, data_loader, num_samples=6, save_to_file=False, file_prefix="prediction"):
    model.eval()
    num_batches = len(data_loader)
    batch_size = data_loader.batch_size
    
    all_images = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.to('cuda:0')
            labels = labels.to('cuda:0')
            
            outputs = model(inputs).squeeze()
            preds = torch.sigmoid(outputs) > 0.5
            
            all_images.extend(inputs.cpu())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if not save_to_file:
                # Display images and predictions
                fig = plt.figure(figsize=(15, 6))
                for idx in range(min(num_samples, batch_size)):
                    if idx < len(inputs):
                        ax = fig.add_subplot(2, 4, idx+1, xticks=[], yticks=[])
                        imshow(inputs[idx])
                        ax.set_title(f'Predicted: {preds[idx].item()}\nActual: {labels[idx].item()}')
                plt.show()
            else:
                # Save images and predictions to file
                for idx in range(len(inputs)):
                    plt.figure()
                    imshow(inputs[idx])
                    plt.title(f'Predicted: {preds[idx].item()}\nActual: {labels[idx].item()}')
                    plt.savefig(f'./predicted/{file_prefix}_batch{batch_idx}_img{idx}.png')
                    plt.close()

    return all_images, all_preds, all_labels


def imshow(inp, title=None):
    inp = inp.cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

# Create a checkpoint callback to save the best model
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_top_k=1,
    save_last=True,
)

# Create a logger for TensorBoard
logger = TensorBoardLogger("tb_logs", name="cancer_classifier")

# Learning rate monitor
lr_monitor = LearningRateMonitor(logging_interval='epoch')

# Initialize the trainer
trainer = pl.Trainer(max_epochs=100, accelerator="gpu", devices=1, callbacks=[checkpoint_callback, lr_monitor], logger=logger)

model = CancerClassifier()

if os.path.exists('best_model.pth'):
    print("Loading model from best_model.pth")
    model.load_state_dict(torch.load('best_model.pth'))
else:
    print("Training the model")
    # Train the model
    trainer.fit(model, train_loader, val_loader)
    # Save the model
    torch.save(model.state_dict(), 'best_model.pth')

# Load the best model
model.load_state_dict(torch.load('best_model.pth'))
model = model.to('cuda:0')

# Put the model in evaluation mode
model.eval()

all_preds = []
all_labels = []

# Perform evaluation on the validation set
val_accuracy = torchmetrics.Accuracy(task="binary").to('cuda:0')
val_loss = 0.0
criterion = nn.BCEWithLogitsLoss()
torch.cuda.empty_cache()
with torch.no_grad():
    for batch in val_loader:
        images, labels = batch
        images = images.to("cuda:0")  # Move images to the GPU
        labels = labels.to("cuda:0")  # Move labels to the GPU
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels.float())
        val_loss += loss.item()
        preds = torch.sigmoid(outputs) > 0.5
        val_accuracy.update(preds.float(), labels.float())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

val_loss /= len(val_loader)
print(f'Validation Loss: {val_loss}')
print(f'Validation Accuracy: {val_accuracy.compute()}')

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

conf_matrix = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:") 
print(conf_matrix)

# Visualize predictions on validation set
#visualize_predictions(model, val_loader)

# Call the function with the test data path
test_data_path = './Data/test'
all_preds, all_labels = test_model(model, test_data_path)

# Visualize predictions on test set
test_dataset = torchvision.datasets.ImageFolder(root=test_data_path, transform=val_transforms)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
visualize_predictions(model, test_loader, num_samples=11, save_to_file=True)

# Print some predictions for review
print("Predictions:", all_preds)
all_labels_str = [False if label == 0 else True for label in all_labels]

# Print actual labels
print("Actual Labels:", all_labels_str)
