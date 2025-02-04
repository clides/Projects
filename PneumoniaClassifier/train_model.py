import os

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import accuracy_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PneumoniaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Initialize the PneumoniaDataset with the directory of images and optional transformations.

        Parameters:
        root_dir (str): The root directory containing the image data.
        transform (callable, optional): Optional transform to be applied on an image.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []  # List to store paths to all images
        self.labels = []       # List to store labels corresponding to the images
        
        # Iterate over the classes, 'NORMAL' and 'PNEUMONIA'
        for label in ['NORMAL', 'PNEUMONIA']:
            class_dir = os.path.join(root_dir, label)  # Get directory for each class
            
            # Iterate over all images in the class directory
            for img_name in os.listdir(class_dir):
                # Append the full path of the image to image_paths
                self.image_paths.append(os.path.join(class_dir, img_name))
                # Append the label (0 for NORMAL, 1 for PNEUMONIA) to labels
                self.labels.append(0 if label == 'NORMAL' else 1)
                
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
# transform needed for ResNet18
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# setting up all the datasets
train_dataset = PneumoniaDataset(root_dir='data/train', transform=transform)
test_dataset = PneumoniaDataset(root_dir='data/test', transform=transform)
val_dataset = PneumoniaDataset(root_dir='data/val', transform=transform)

# sending all datasets to DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)
val_loader = DataLoader(val_dataset, batch_size=32)

# setting up the ResNet18 model and configuring it so the full connected layer has 2 outputs
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 2) # 2 outputs for NORMAL and PNEUMONIA
model = model.to(device)

# creating loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training loop to fine tune the model
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # 1. Forward pass
        outputs = model(images)
        
        # 2. Compute loss
        loss = loss_fn(outputs, labels)
        
        # 3. Zero-grad
        optimizer.zero_grad()
        
        # 4. Backpropagation
        loss.backward()
        
        # 5. Gradient descent
        optimizer.step()
        
        running_loss += loss.item()
        
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")
    
    # computing the validation accuracy
    model.eval()
    val_labels = []
    val_preds = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            val_labels.extend(labels.numpy())
            val_preds.extend(preds.numpy())
    
    val_accuracy = accuracy_score(val_labels, val_preds)
    print(f"Validation accuracy: {val_accuracy}")
    
    
# testing the model
model.eval()
test_labels = []
test_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        test_labels.extend(labels.numpy())
        test_preds.extend(preds.numpy())

test_accuracy = accuracy_score(test_labels, test_preds)
print(f"Test accuracy: {test_accuracy}")

# saving the model weights
torch.save(model.state_dict(), 'pneumonia_model.pth')