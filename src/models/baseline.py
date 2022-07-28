import torch.nn as nn
import torch.nn.functional as F

class BaselineCNN(nn.Module):
    def __init__(self, num_classes):
        super(BaselineCNN, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
        # self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        # self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.fc1 = nn.LazyLinear(128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):

        # out = F.max_pool2d(F.relu(self.conv1(x)), 2)
        # out = F.max_pool2d(F.relu(self.conv2(out)), 2)
        # out = F.max_pool2d(F.relu(self.conv3(out)), 2)

        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.max_pool1(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.max_pool2(out))
        
        out = out.reshape(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out