#Uses image folder instead of custom dataset. 

import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder


transform = T.Compose([
  T.Resize(256),
  T.RandomCrop(224),
  T.RandomHorizontalFlip(),
  T.RandomVerticalFlip(),
  T.ToTensor()   #Required!
])

dataset = ImageFolder('train_crc', transform=transform)
train_data_len = int(len(dataset)*0.8)
valid_data_len = int(len(dataset) - train_data_len)
train_data, val_data = random_split(dataset, [train_data_len, valid_data_len])
print(len(train_data), len(val_data))
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=True)



