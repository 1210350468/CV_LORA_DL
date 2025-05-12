import torch
from torch import nn


seed = 10001
torch.manual_seed(seed)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.conv = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5, stride=1)
        self.bn = nn.BatchNorm2d(num_features=10, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(in_features=10, out_features=1)

    def forward(self, x):
        x = self.conv(x)
        var, mean = torch.var_mean(x, dim=[0, 2, 3])
        print("x's mean: {}\nx's var: {}".format(mean.detach().numpy(), var.detach().numpy()))
        
        x = self.bn(x)
        print('-----------------------------------------------------------------------------------')
        print("x's mean: {}\nx's var: {}".format(self.bn.running_mean.numpy(), self.bn.running_var.numpy()))
        
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        output = self.linear(x)

        return output


if __name__ == '__main__':
    model = MyModel()

    inputs = torch.randn(size=(128, 3, 32, 32))
    model(inputs)
