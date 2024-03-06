import torch
import torch.nn.functional as F

class LeNet(torch.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 6, kernel_size=5)
        self.pool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = torch.nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        # Adjust the input size for the first fully connected layer
        self.fc1 = torch.nn.Linear(16 * 4 * 4, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = F.relu(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)  # Apply softmax along dimension 1
        return x