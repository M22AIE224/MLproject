import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_recall_curve, average_precision_score
import numpy as np
import torch.nn.functional as F

train_dataset = datasets.USPS(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.USPS(root='./data', train=False, download=True, transform=transforms.ToTensor())

if(torch.cuda.is_available()):
  device = "cuda"
else:
  device = "cpu"

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
class Model_CNN_v1(nn.Module):
    def __init__(self):
        super(Model_CNN_v1, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class Model_CNN_v2(nn.Module):
    def __init__(self):
        super(Model_CNN_v2, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(-1, 32 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
class Model_CNN_v3(nn.Module):
    def __init__(self):
        super(Model_CNN_v3, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 32 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    
def train(model, train_loader, test_loader, criterion, optimizer, num_epochs, device,writer):
    train_acc_history = []
    test_acc_history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_acc = evaluate(model, train_loader, device)
        test_acc = evaluate(model, test_loader, device)

        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)


        train_acc, train_predictions, train_labels = evaluate(model, train_loader, device)
        test_acc, test_predictions, test_labels = evaluate(model, test_loader, device)

        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/test', test_acc, epoch)

        for class_idx in range(10):
            train_precision, train_recall, _ = precision_recall_curve(train_labels == class_idx, train_predictions[:, class_idx])
            train_average_precision = average_precision_score(train_labels == class_idx, train_predictions[:, class_idx])

            test_precision, test_recall, _ = precision_recall_curve(test_labels == class_idx, test_predictions[:, class_idx])
            test_average_precision = average_precision_score(test_labels == class_idx, test_predictions[:, class_idx])

            # Log precision-recall curve and average precision for each class
            writer.add_pr_curve(f'Precision-Recall/Class_{class_idx}/train', train_labels == class_idx, train_predictions[:, class_idx], epoch)
            writer.add_pr_curve(f'Precision-Recall/Class_{class_idx}/test', test_labels == class_idx, test_predictions[:, class_idx], epoch)

            writer.add_scalar(f'Average Precision/Class_{class_idx}/train', train_average_precision, epoch)
            writer.add_scalar(f'Average Precision/Class_{class_idx}/test', test_average_precision, epoch)


        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    return train_acc_history, test_acc_history

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_predictions.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    return correct / total, all_predictions, all_labels
writer = SummaryWriter()
#experiment with 3 models 
for version in ['v1','v2','v3']:
    if version == 'v1':
        model = Model_CNN_v1().to(device)
    elif version == 'v2':
        model = Model_CNN_v2().to(device)
    elif version == 'v3':
        model = Model_CNN_v3().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    cnn_train_acc_history, cnn_test_acc_history = train(model, train_loader, test_loader, criterion, optimizer, 10, device,writer)
    evaluate(model, test_loader, device)

writer.close()
