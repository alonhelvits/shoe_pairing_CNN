import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def ID1():
    '''
        Personal ID of the first student.
    '''
    # Insert your ID here
    return 000000000

def ID2():
    '''
        Personal ID of the second student. Fill this only if you were allowed to submit in pairs, Otherwise leave it zeros.
    '''
    # Insert your ID here
    return 000000000

class CNN(nn.Module):
    def __init__(self): 
        super(CNN, self).__init__()
        # Parameters
        self.n = 8
        self.kernel = 5
        self.padding = int((self.kernel - 1) / 2.)
        self.stride = 2

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.n, kernel_size=self.kernel, padding=self.padding)
        self.conv2 = nn.Conv2d(self.n, 2 * self.n, kernel_size=self.kernel, padding=self.padding)
        self.conv3 = nn.Conv2d(2 * self.n, 4 * self.n, kernel_size=self.kernel, padding=self.padding)
        self.conv4 = nn.Conv2d(4 * self.n, 8 * self.n, kernel_size=self.kernel, padding=self.padding)

        # Fully connected layers
        self.linear1 = nn.Linear(8 * self.n * 28 * 14, 100)
        self.linear2 = nn.Linear(100, 2)

        # Dropout layers
        self.drop1 = nn.Dropout(0.05)
        self.drop2 = nn.Dropout(0.8)

        # Batch Normalization layers
        self.batch_norm1 = nn.BatchNorm2d(self.n)
        self.batch_norm2 = nn.BatchNorm2d(2 * self.n)
        self.batch_norm3 = nn.BatchNorm2d(4 * self.n)
        self.batch_norm4 = nn.BatchNorm2d(8 * self.n)
        self.batch_norm5 = nn.BatchNorm1d(100)

    def forward(self, inp):
        '''
        Forward pass of the CNN model.

        Parameters:
        - inp: Input image (pytorch tensor) with shape (N, 3, 448, 224).

        Returns:
        - out: Output tensor with shape (N, 2) representing same/different pairs.
        '''
        out = self.conv1(inp)
        out = self.batch_norm1(out)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)

        out = self.conv2(out)
        out = self.batch_norm2(out)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)

        out = self.conv3(out)
        out = self.batch_norm3(out)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)

        out = self.conv4(out)
        #out = self.batch_norm4(out)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)

        out = out.reshape(-1, 8 * self.n * 28 * 14)

        #out = self.drop1(out)
        out = self.linear1(out)
        #out = self.batch_norm5(out)
        out = F.relu(out)
        out = self.drop2(out)
        out = self.linear2(out)

        return out

class CNN2(nn.Module):
    def __init__(self): 
        super(CNN2, self).__init__()
        # Parameters
        self.n = 32
        self.kernel = 5
        self.padding = int((self.kernel - 1) / 2.)
        self.stride = 2

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.n, kernel_size=self.kernel, padding=self.padding)
        self.conv2 = nn.Conv2d(self.n, 2 * self.n, kernel_size=self.kernel, padding=self.padding)
        self.conv3 = nn.Conv2d(2 * self.n, 4 * self.n, kernel_size=self.kernel, padding=self.padding)
        self.conv4 = nn.Conv2d(4 * self.n, 8 * self.n, kernel_size=self.kernel, padding=self.padding)

        # Fully connected layers
        self.linear1 = nn.Linear(8 * self.n * 28 * 14, 100)
        self.linear2 = nn.Linear(100, 2)

        # Dropout layers
        self.drop2 = nn.Dropout(0.7)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Batch Normalization layers
        self.batch_norm1 = nn.BatchNorm2d(self.n)
        self.batch_norm2 = nn.BatchNorm2d(2 * self.n)
        self.batch_norm3 = nn.BatchNorm2d(4 * self.n)
        self.batch_norm4 = nn.BatchNorm2d(8 * self.n)
        self.batch_norm5 = nn.BatchNorm1d(100)

    def forward(self, inp):
        '''
        Forward pass of the CNN model.

        Parameters:
        - inp: Input image (pytorch tensor) with shape (N, 3, 448, 224).

        Returns:
        - out: Output tensor with shape (N, 2) representing same/different pairs.
        '''
        out = self.conv1(inp)
        #out = self.batch_norm1(out)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)

        out = self.conv2(out)
        #out = self.batch_norm2(out)
        out = F.relu(out)
        out = self.max_pool(out)

        out = self.conv3(out)
        #out = self.batch_norm3(out)
        out = F.relu(out)
        out = self.max_pool(out)

        out = self.conv4(out)
        #out = self.batch_norm4(out)
        out = F.relu(out)
        out = self.max_pool(out)

        out = out.reshape(-1, 8 * self.n * 28 * 14)

        out = self.drop2(out)
        out = self.linear1(out)
        #out = self.batch_norm5(out)
        out = F.relu(out)
        out = self.linear2(out)

        return out

class CNNChannel(nn.Module):
    def __init__(self):
        super(CNNChannel, self).__init__()

        self.n = 64
        self.kernel_size = 7
        self.padding_size = (self.kernel_size - 1) // 2

        self.conv1 = nn.Conv2d(in_channels=6, out_channels=self.n, kernel_size=self.kernel_size, padding=self.padding_size)
        self.conv2 = nn.Conv2d(in_channels=self.n, out_channels=2 * self.n, kernel_size=self.kernel_size, padding=self.padding_size)
        self.conv3 = nn.Conv2d(in_channels=2 * self.n, out_channels=4 * self.n, kernel_size=self.kernel_size, padding=self.padding_size)
        self.conv4 = nn.Conv2d(in_channels=4 * self.n, out_channels=8 * self.n, kernel_size=self.kernel_size, padding=self.padding_size)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.7)

        self.flattened_size = 8 * self.n * 14 * 14

        self.fc1 = nn.Linear(self.flattened_size, 100)
        self.fc2 = nn.Linear(100, 2)

        # Batch Normalization layers
        self.batch_norm1 = nn.BatchNorm2d(self.n)
        self.batch_norm2 = nn.BatchNorm2d(2 * self.n)
        self.batch_norm3 = nn.BatchNorm2d(4 * self.n)
        self.batch_norm4 = nn.BatchNorm2d(8 * self.n)

    def forward(self, inp):
        inp_reshaped = inp.reshape(inp.size(0), 2 * inp.size(1), inp.size(2) // 2, inp.size(3))

        out = self.conv1(inp_reshaped)
        out = self.batch_norm1(out)
        out = F.relu(out)
        out = self.max_pool(out)

        out = self.conv2(out)
        out = self.batch_norm2(out)
        out = F.relu(out)
        out = self.max_pool(out)

        out = self.conv3(out)
        out = self.batch_norm3(out)
        out = F.relu(out)
        out = self.max_pool(out)

        out = self.conv4(out)
        out = self.batch_norm4(out)
        out = F.relu(out)
        out = self.max_pool(out)

        #out = out.view(out.size(0), -1)

        out = out.view(-1, self.flattened_size)

        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)

        return out