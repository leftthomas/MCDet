import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnext50_32x4d


class Model(nn.Module):
    def __init__(self, ensemble_size, meta_class_size, model_type):
        super(Model, self).__init__()

        # backbone
        backbones = {'resnet18': (resnet18, 1), 'resnet34': (resnet34, 1), 'resnet50': (resnet50, 4),
                     'resnext50_32x4d': (resnext50_32x4d, 4)}
        backbone, expansion = backbones[model_type]

        # configs
        self.ensemble_size = ensemble_size

        # common features
        common_module_names = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1']
        self.common_extractor, basic_model = [], backbone(True)
        for name, module in basic_model.named_children():
            if name in common_module_names:
                self.common_extractor.append(module)
        self.common_extractor = nn.Sequential(*self.common_extractor)
        print("# trainable common feature parameters:",
              sum(param.numel() if param.requires_grad else 0 for param in self.common_extractor.parameters()))

        # individual features
        self.individual_extractor, individual_module_names = [], ['layer2', 'layer3', 'layer4']
        for i in range(ensemble_size):
            heads, basic_model = [], backbone(True)
            for name, module in basic_model.named_children():
                if name in individual_module_names:
                    heads.append(module)
            heads = nn.Sequential(*heads)
            self.individual_extractor.append(heads)
        self.individual_extractor = nn.ModuleList(self.individual_extractor)
        print("# trainable individual feature parameters:",
              sum(param.numel() if param.requires_grad else 0 for param in
                  self.individual_extractor.parameters()) // ensemble_size)

        # individual classifiers
        self.classifiers = nn.ModuleList([nn.Linear(512 * expansion, meta_class_size) for _ in range(ensemble_size)])
        print("# trainable individual classifier parameters:",
              sum(param.numel() if param.requires_grad else 0 for param in
                  self.classifiers.parameters()) // ensemble_size)

    def forward(self, x):
        batch_size = x.size(0)
        common_feature = self.common_extractor(x)
        out = []
        for i in range(self.ensemble_size):
            individual_feature = self.individual_extractor[i](common_feature)
            global_feature = F.adaptive_avg_pool2d(individual_feature, output_size=(1, 1)).view(batch_size, -1)
            classes = self.classifiers[i](global_feature)
            out.append(classes)
        out = torch.stack(out, dim=1)
        return out
