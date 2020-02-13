import torch
import torch.nn as nn
from torch.nn import functional as F

from backbone import FastResNet


class Model(nn.Module):
    def __init__(self, ensemble_size, meta_class_size):
        super(Model, self).__init__()

        # configs
        self.ensemble_size = ensemble_size

        # common features
        self.head = FastResNet(pretrained=True).head
        self.body = FastResNet(pretrained=True).body
        print("# trainable common feature parameters:",
              sum(param.numel() if param.requires_grad else 0 for param in self.head.parameters())
              + sum(param.numel() if param.requires_grad else 0 for param in self.body.parameters()))

        # individual features
        self.tails = []
        for i in range(ensemble_size):
            self.tails.append(FastResNet(pretrained=True).tail)
        self.tails = nn.ModuleList(self.tails)
        print("# trainable individual feature parameters:",
              sum(param.numel() if param.requires_grad else 0 for param in
                  self.tails.parameters()) // ensemble_size)

        self.classifier = nn.ModuleList([nn.Linear(512, meta_class_size) for _ in range(ensemble_size)])

    def forward(self, x):
        batch_size = x.size(0)
        shared = self.body(self.head(x))
        out = []
        for i in range(self.ensemble_size):
            feature = self.tails[i](shared)
            feature = F.adaptive_avg_pool2d(feature, output_size=(1, 1)).view(batch_size, -1)
            feature = self.classifier[i](feature)
            out.append(feature)
        out = torch.stack(out, dim=1)
        return out
