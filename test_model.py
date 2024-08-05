import torch
import torchvision.transforms as T
from torchvision.models import resnet18
from torchvision.datasets.folder import default_loader
import torch.nn as nn
import os

# Load the model
model = resnet18()
model.fc = nn.Linear(in_features=512, out_features=1)
model.load_state_dict(torch.load('model.pth'))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

test_transform = T.Compose([ 
    T.Resize(256),
    T.RandomCrop(224),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ToTensor()   #Required!
]) 
# Path to the directory containing test images
test_dir = 'Test set 01'

image_paths = [os.path.join(test_dir, img) for img in os.listdir(test_dir)]

# Function to load and preprocess images
def load_image(image_path):
    image = default_loader(image_path)
    image = test_transform(image)
    return image

def make_predictions(image_paths, model):
    model.eval()
    predictions = []

    with torch.no_grad():
        for img_path in image_paths:
            image = load_image(img_path).unsqueeze(0).to(device)
            output = model(image)
            print(output)   # output = tensor([[-112.8091]]) form
            pred = torch.sigmoid(output[:, 0]).round().cpu().numpy()[0]  #The rounding of the tensor output to 0 or 1 
                                                                        #works by first applying the sigmoid function to the
                                                                        #  model's output and then using the round() method on 
                                                                        # the resulting tensor.
            
            predictions.append((img_path, int(pred)))

    return predictions

# Make predictions on the test dataset
test_predictions = make_predictions(image_paths, model)

# Print the filenames and their predictions
for img_path, prediction in test_predictions:
    filename = os.path.basename(img_path)
    print(f"Filename: {filename}, Prediction: {prediction}")
