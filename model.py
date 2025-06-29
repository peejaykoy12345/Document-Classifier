import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.functional import F
from PIL import Image

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),                  
    transforms.ToTensor()                   
])

dataset = ImageFolder("DataSet", transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  
        x = self.pool(torch.relu(self.conv2(x))) 
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
model = CNN()

criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 

epochs = 100

for epoch in range(epochs):
    for images, labels in loader:
        optimizer.zero_grad()  
        outputs = model(images)  
        loss = criterion(outputs, labels)  
        loss.backward()  
        optimizer.step()  

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

correct = 0
total = 0
with torch.no_grad():
    for images, labels in loader:
        outputs = model(images)
        predicted = torch.argmax(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    print(f'Accuracy of the model on the dataset: {100 * correct / total:.2f}%')

ID_TEST_image_path = "TestSet/ID_TESTING_2.png"
ID_TEST_image = Image.open(ID_TEST_image_path).convert("L")  
ID_TEST_image = transform(ID_TEST_image)  
ID_TEST_image = ID_TEST_image.unsqueeze(0)

RECEIPT_TEST_image_path = "TestSet/RECEIPT_TESTING_2.png"
RECEIPT_TEST_image = Image.open(RECEIPT_TEST_image_path).convert("L")  
RECEIPT_TEST_image = transform(RECEIPT_TEST_image)  
RECEIPT_TEST_image = RECEIPT_TEST_image.unsqueeze(0)

classes = dataset.classes

with torch.no_grad():
    ID_TEST_PREDICTION = model(ID_TEST_image)
    RECEIPT_TEST_PREDICTION = model(RECEIPT_TEST_image)
    ID_TEST_output = torch.argmax(ID_TEST_PREDICTION, dim=1)
    RECEIPT_TEST_output = torch.argmax(RECEIPT_TEST_PREDICTION, dim=1)

    #print(f"ID_TEST Prediction: {ID_TEST_PREDICTION}")
    #print(f"RECEIPT_TEST Prediction: {RECEIPT_TEST_PREDICTION}")
    print(f"ID_TEST Output: {classes[ID_TEST_output]}")
    print(f"RECEIPT_TEST Output: {classes[RECEIPT_TEST_output]}")
 