import torch.nn as nn
# from mygqcnn.graspDataLoader import GraspDataset
import torch
from torchsummary import summary

class GQCNN(nn.Module):
    def __init__(self):
        super(GQCNN, self).__init__()
        self.layers = self.structure()
        # self.lossWeight = torch.ones(32).cuda()
        # self.lossWeight[1::2] = 1200
        self.sf = nn.Softmax(dim=1)
        self.ceLoss = nn.CrossEntropyLoss(weight=torch.tensor([10., 1.]))

    def forward(self, input_tensor):
        return self.layers(input_tensor)

    @staticmethod
    def structure():
        layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=9, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=9, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 128, kernel_size=14, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            # nn.Softmax(dim=1)
            nn.Conv2d(128, 64, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=1, padding=0),
            # nn.ReLU(inplace=True),
            # nn.Sigmoid()
        )

        return layers

    def compute_loss(self, input_tensor, metric):

        label_pred = self(input_tensor)
        label_pred = label_pred.squeeze(2).squeeze(2)
        loss = self.ceLoss(label_pred, metric)
        label_pred = torch.where(self.sf(label_pred) > 0.5)[1]

        return loss, label_pred

    # cross_entropy version
    def get_label(self, input_tensor):
        output_tensor = self(input_tensor).squeeze(2).squeeze(2)
        label_pred = torch.where(self.sf(output_tensor) > 0.5)[1]

        return label_pred


if __name__ == '__main__':
    cnn = GQCNN().cuda()
    summary(cnn, (1, 96, 96), batch_size=8)








