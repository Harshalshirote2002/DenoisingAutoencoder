from torch import nn

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2))

        # Representation here is (7, 7, 32)

        self.conv1UpScale = nn.Conv2d(32, 32, kernel_size=(3,3), padding=1)
        self.act1UpScale = nn.ReLU()
        self.pool1UpScale = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv2UpScale = nn.Conv2d(32, 32, kernel_size=(3,3), padding=1)
        self.act2UpScale = nn.ReLU()
        self.pool2UpScale = nn.Upsample(scale_factor=2, mode='nearest')

        self.decoded = nn.Conv2d(32, 1, kernel_size=(3, 3), padding=1)
        self.decodedAct = nn.Sigmoid()

    def forward(self, x):
        # Input (28, 28, 1)
        x = self.act1(self.conv1(x))
        x = self.pool1(x)
        
        x = self.act2(self.conv2(x))
        x = self.pool2(x)

        x = self.act1UpScale(self.conv1UpScale(x))
        x = self.pool1UpScale(x)

        x = self.act2UpScale(self.conv2UpScale(x))
        x = self.pool2UpScale(x)
        x = self.decodedAct(self.decoded(x))

        return x
