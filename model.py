import torch
import torch.nn as nn

from backbone import FastSCNN


class Model(nn.Module):
    def __init__(self, ensemble_size, meta_class_size):
        super(Model, self).__init__()

        # configs
        self.ensemble_size = ensemble_size

        # common features
        basic_model = FastSCNN(in_channels=3, num_classes=meta_class_size)
        self.learning_to_down_sample = basic_model.learning_to_down_sample
        self.global_feature_extractor = basic_model.global_feature_extractor
        self.feature_fusion = basic_model.feature_fusion
        print("# trainable common feature parameters:",
              sum(param.numel() if param.requires_grad else 0 for param in self.learning_to_down_sample.parameters())
              + sum(param.numel() if param.requires_grad else 0 for param in self.global_feature_extractor.parameters())
              + sum(param.numel() if param.requires_grad else 0 for param in self.feature_fusion.parameters()))

        # individual features
        self.individual_extractor = []
        for i in range(ensemble_size):
            self.individual_extractor.append(FastSCNN(in_channels=3, num_classes=meta_class_size).classifier)
        self.individual_extractor = nn.ModuleList(self.individual_extractor)
        print("# trainable individual feature parameters:",
              sum(param.numel() if param.requires_grad else 0 for param in
                  self.individual_extractor.parameters()) // ensemble_size)

    def forward(self, x):
        shared = self.learning_to_down_sample(x)
        x = self.global_feature_extractor(shared)
        common_feature = self.feature_fusion(shared, x)
        out = []
        for i in range(self.ensemble_size):
            individual_feature = self.individual_extractor[i](common_feature)
            out.append(individual_feature)
        out = torch.stack(out, dim=1)
        return out
