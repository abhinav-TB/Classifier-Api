import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
import io
from PIL import Image
import torchvision.transforms as transforms


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)
       

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    


model = Net()

model.load_state_dict(torch.load('cifar_net.pth', map_location=torch.device('cpu')))
model.eval()


def transform_image(img_bytes):
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32,32)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = Image.open(io.BytesIO(img_bytes))
    return transform(image).unsqueeze(0)

def get_predictions(image_tensor):
    images = image_tensor
    print(images.shape)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    return predicted
