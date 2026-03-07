import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleCNN(nn.Module):

    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(64*8*8, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):

        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x


model = SimpleCNN().to(device)
model.load_state_dict(torch.load("models/cifar_cnn.pth"))
model.eval()

# clean dataset testing

clean_images = torch.load("datasets/cifar_clean.pt").to(device)
labels = torch.load("datasets/cifar_labels.pt").to(device)

correct = 0

for i in range(len(clean_images)):

    image = clean_images[i].unsqueeze(0)

    output = model(image)

    pred = output.argmax(dim=1)

    if pred == labels[i]:
        correct += 1

accuracy_clean = correct / len(clean_images)

print("Accuracy on clean images:", accuracy_clean)

# adversarial images testing

adv_images = torch.load("datasets/cifar_adv.pt").to(device)
labels = torch.load("datasets/cifar_labels.pt").to(device)

correct = 0

for i in range(len(adv_images)):

    image = adv_images[i].unsqueeze(0)

    output = model(image)

    pred = output.argmax(dim=1)

    if pred == labels[i]:
        correct += 1

accuracy_adversarial = correct / len(adv_images)

print("Accuracy on adversarial images:", accuracy_adversarial)