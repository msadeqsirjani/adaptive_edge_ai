import torch
import torch.nn as nn

class DynamicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, width_multiplier=1.0):
        super().__init__()
        self.width_multiplier = width_multiplier
        scaled_channels = int(out_channels * width_multiplier)
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, scaled_channels, 3, padding=1),
            nn.BatchNorm2d(scaled_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)

class DynamicNeuralNetwork(nn.Module):
    def __init__(self, input_channels=3, num_classes=10, complexity_levels=[0.25, 0.5, 1.0]):
        super().__init__()
        self.complexity_levels = complexity_levels
        self.current_complexity = complexity_levels[-1]
        
        # Base architecture
        self.features = nn.ModuleList([
            DynamicBlock(input_channels, 32, self.current_complexity),
            DynamicBlock(32, 64, self.current_complexity),
            DynamicBlock(64, 128, self.current_complexity)
        ])
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )
        
    def adjust_complexity(self, target_complexity):
        """Dynamically adjust the network complexity"""
        if target_complexity in self.complexity_levels:
            self.current_complexity = target_complexity
            for block in self.features:
                block.width_multiplier = target_complexity
                
    def forward(self, x):
        for block in self.features:
            x = block(x)
        return self.classifier(x)