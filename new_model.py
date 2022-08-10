import torch
from torchvision import models
from torch import nn

# model identy
new_module = models.resnet50(weights='IMAGENET1K_V2')
new_module.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
new_module.fc = nn.Linear(2048, 4096)
# print(new_module)
# torch.save(new_module, 'new_module.pth')

class MatchNet(nn.Module):
    def __init__(self, new_module):
        super(__class__, self).__init__()
        self.input_1 = new_module
        self.input_2 = new_module

        self.matric_network = nn.Sequential(
            nn.Linear(in_features=8192, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        """
        x has shape 2, N, H, W, 1, where 2 means two patches
        """
        x1 = x[:,:,:, 0:64]  
        x2 = x[:,:,:, 64:128]
        feature_1 = self.input_1(x1)
        feature_2 = self.input_2(x2)

        # test
        #print("features.shape:{}".format(feature_1.cpu().shape))

        res = self.matric_network(torch.cat((feature_1, feature_2), 1))

        # test
        #print("features.shape:{}".format(features.cpu().shape))

        return res

