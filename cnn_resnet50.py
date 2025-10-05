import torch.nn as nn


class Bottleneck(nn.Module):
    """
    Purpose:
    Bottleneck is a building block for ResNet. It is used to increase the depth of the network.
    
    Parameters:
    - in_channels(int): Number of input channels
    - out_channels(int): Number of output channels
    - stride(int): Stride of the first convolutional layer
    - downsample(nn.Sequential): Downsample layer to match the dimensions of the residual
    - kernel_size(int): Size of the convolutional kernel

    Result:
    - out(torch.Tensor): Output tensor after applying the bottleneck
    
    Howto:
    - Apply the first convolutional layer
    - Apply the batch normalization
    - Apply the ReLU activation
    - Apply the second convolutional layer
    - Apply the batch normalization
    - Apply the ReLU activation
    - Apply the third convolutional layer
    - Apply the batch normalization
    - Apply the downsample layer if needed
    - Add the residual
    - Apply the ReLU activation
    """
    expansion = 4 

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, kernel_size=3):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        # Downsample layer
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        """
        Purpose: Defines the computation performed at every call.

        Parameters:
        - x (Tensor): The input tensor.

        Result:
        Returns the output tensor after passing through the bottleneck block.

        Howto:
        - Call this method with the input tensor as the argument.
        - The method will pass the input through each layer of the network.
        """
        # Save the residual
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        # Apply the downsample layer if needed
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """
    Purpose: ResNet for image classification.

    Parameters:
    - block(nn.Module): The building block for the network
    - layers(list): Number of blocks in each layer
    - num_classes(int): Number of classes to classify
    - kernel_size(int): Size of the convolutional kernel

    Result:
    - out(torch.Tensor): Output tensor after applying the ResNet

    Howto:
    - Call the forward method with the input tensor.
    - The method will first apply a convolutional layer to the input tensor.
    - Then, it will apply batch normalization to standardize the output from the convolutional layer.
    - The method will continue to pass the tensor through the rest of the network layers.
    - Finally, it will return the output tensor, which represents the network's predictions.
    """
    def __init__(self, block, layers, num_classes=2, kernel_size=3):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv = nn.Conv2d(in_channels = 6, out_channels = 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(block, out_channels = 64, blocks = layers[0], kernel_size=kernel_size)
        self.layer2 = self.make_layer(block, out_channels = 128, blocks = layers[1], stride=2, kernel_size=kernel_size)
        self.layer3 = self.make_layer(block, out_channels = 256, blocks = layers[2], stride=2, kernel_size=kernel_size)
        self.layer4 = self.make_layer(block, out_channels = 512, blocks = layers[3], stride=2, kernel_size=kernel_size)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1, kernel_size=3):
        """
        Purpose: Creates a layer of the network.

        Parameters:
        - block(nn.Module): The building block for the network
        - out_channels(int): Number of output channels
        - blocks(int): Number of blocks in the layer
        - stride(int): Stride of the first convolutional layer
        - kernel_size(int): Size of the convolutional kernel
        
        Result:
        - out(nn.Sequential): The layer of the network
        
        Howto:
        - Call this method with the block, out_channels, blocks, stride, and kernel_size as arguments.
        - The method will create the first block of the layer.
        - Then, it will create the rest of the blocks in the layer.
        - It will return the layer.
        """
        downsample = None
        # Check if the input channels and output channels are the same
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        # The first block of the layer
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, kernel_size=kernel_size))
        self.in_channels = out_channels * block.expansion
        # The rest of the blocks
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, kernel_size=kernel_size))
        return nn.Sequential(*layers)

    # Forward pass
    def forward(self, x):
        """
        Purpose: Defines the computation performed at every call.

        Parameters:
        - x (Tensor): The input tensor.
        
        Result:
        Returns the output tensor after passing through the ResNet.

        Howto:
        - Call this method with the input tensor as the argument.
        - The method will pass the input through each layer of the network.
        - It will return the output tensor, which represents the network's predictions.
        """
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return out

def resnet50(kernel_size):
    return ResNet(Bottleneck, [3, 4, 6, 3], kernel_size) 
