import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
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

transform = transforms.Compose([transforms.ToTensor()])

testset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# Take first 1000 images
subset_indices = torch.randperm(len(testset))[:1000]
subset = torch.utils.data.Subset(testset, subset_indices)

loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False)

epsilon = 0.01

criterion = nn.CrossEntropyLoss()

adv_images = []
clean_images = []
labels_list = []

for image, label in loader:

    image = image.to(device)
    label = label.to(device)

    # Save clean image BEFORE perturbation
    clean_images.append(image.detach().cpu())

    image.requires_grad = True

    output = model(image)

    loss = criterion(output, label)

    model.zero_grad()

    loss.backward()

    grad = image.grad.data

    adv = image + epsilon * grad.sign()

    adv = torch.clamp(adv, 0, 1)

    adv_images.append(adv.detach().cpu())
    labels_list.append(label.cpu())

# Convert lists to tensors
adv_images = torch.cat(adv_images)
clean_images = torch.cat(clean_images)
labels = torch.cat(labels_list)

# Save datasets
torch.save(clean_images, "datasets/cifar_clean.pt")
torch.save(adv_images, "datasets/cifar_adv.pt")
torch.save(labels, "datasets/cifar_labels.pt")

print("Clean dataset saved:", clean_images.shape)
print("Adversarial dataset saved:", adv_images.shape)