import torch
import torch.nn as nn
from torch.nn import functional as F

from backbone import eca_mobilenet_v2


class Model(nn.Module):
    def __init__(self, ensemble_size, meta_class_size, feature_dim, share_type='none'):
        super(Model, self).__init__()

        # configs
        self.ensemble_size = ensemble_size
        module_names = ['conv', 'block1', 'block2', 'block3', 'block4', 'block5', 'block6', 'block7', 'last_conv']

        # common features
        if share_type != 'none':
            common_module_names = module_names[:module_names.index(share_type)]
        else:
            common_module_names = []
        self.head = []
        for name, module in eca_mobilenet_v2(pretrained=True).named_modules():
            if name in common_module_names:
                self.head.append(module)
        self.head = nn.Sequential(*self.head)
        print("# trainable common feature parameters:", sum(param.numel() if param.requires_grad else 0 for
                                                            param in self.head.parameters()))

        # individual features
        self.tails = []
        for i in range(ensemble_size):
            self.tails.append(mobilenet_v2(pretrained=True).features)
        self.tails = nn.ModuleList(self.tails)
        print("# trainable individual feature parameters:",
              sum(param.numel() if param.requires_grad else 0 for param in
                  self.tails.parameters()) // ensemble_size)

        self.classifier = nn.ModuleList([nn.Linear(1280, meta_class_size) for _ in range(ensemble_size)])

    def forward(self, x):
        batch_size = x.size(0)
        shared = self.head(x)
        features, out = [], []
        for i in range(self.ensemble_size):
            feature = self.tails[i](shared)
            feature = F.adaptive_avg_pool2d(feature, output_size=(1, 1)).view(batch_size, -1)
            features.append(feature)
            classes = self.classifier[i](feature)
            out.append(classes)
        features = torch.sum(torch.stack(features, dim=1), dim=1)
        out = torch.stack(out, dim=1)
        return features, out
