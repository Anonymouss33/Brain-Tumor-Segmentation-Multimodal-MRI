import torch
import torch.nn as nn
import torch.nn.functional as F

# Convolution Block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)

# U-Net Encoder
class Encoder(nn.Module):
    def __init__(self, in_channels, base_channels=32):
        super(Encoder, self).__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels*2)
        self.enc3 = ConvBlock(base_channels*2, base_channels*4)
        self.enc4 = ConvBlock(base_channels*4, base_channels*8)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        return x1, x2, x3, x4

# BESNet - Detail Branch
class DetailBranch(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DetailBranch, self).__init__()
        self.detail = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.detail(x)

# BESNet - Semantic Branch
class SemanticBranch(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SemanticBranch, self).__init__()
        self.semantic = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.semantic(x)

# Feature Fusion Module (FFM)
class FeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, detail, semantic):
        # Upsample semantic to match detail size
        semantic = F.interpolate(semantic, size=detail.shape[2:], mode='bilinear', align_corners=True)
        fusion = torch.cat([detail, semantic], dim=1)
        return self.relu(self.bn1(self.conv1(fusion)))

# Full U-Net + BESNet Model
class UNetBESNet(nn.Module):
    def __init__(self, in_channels=4, num_classes=4):
        super(UNetBESNet, self).__init__()
        self.encoder = Encoder(in_channels)
        
        # BESNet branches
        self.detail_branch = DetailBranch(32, 64)
        self.semantic_branch = SemanticBranch(256, 64)
        
        # Feature Fusion
        self.fusion = FeatureFusion(128, 64)
        
        # Final output
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
    
    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        detail = self.detail_branch(x1)
        semantic = self.semantic_branch(x4)
        fusion = self.fusion(detail, semantic)
        out = self.final_conv(fusion)
        return out
