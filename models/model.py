import torch
import torch.nn as nn
from torchvision import models


class DeepfakeDetector(nn.Module):

    def __init__(self, lstm_hidden_size=256, lstm_layers=1, bidirectional=True):
        super(DeepfakeDetector, self).__init__()

        efficientnet = models.efficientnet_b0(pretrained=False)
        self.feature_extractor = efficientnet.features

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.lstm = nn.LSTM(
            input_size=1280,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):

        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        features = self.feature_extractor(x)
        features = self.pool(features)
        features = self.flatten(features)

        features = features.view(B, T, -1)

        lstm_out, _ = self.lstm(features)

        final_feature = lstm_out[:, -1, :]

        return self.classifier(final_feature)