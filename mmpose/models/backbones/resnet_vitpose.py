import torch # type: ignore
import torch.nn as nn # type: ignore
import torchvision.models as models # type: ignore
from ..builder import BACKBONES
from .base_backbone import BaseBackbone
from .vit import ViT

@BACKBONES.register_module()
class ResNetViTPose(BaseBackbone):
    """ResNet backbone with adapter for ViTPose.
    
    This model uses a frozen ResNet50 backbone and adapts its features
    to be compatible with ViTPose's architecture.
    """
    def __init__(self,
                 vitpose_cfg,
                 pretrained_vitpose=None,
                 square_input_size=384,  # Size for ResNet input (will be square)
                 freeze_backbone=True,
                 init_cfg=None):
        super(ResNetViTPose, self).__init__(init_cfg)
        
        # Create and load ResNet backbone
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        
        # Freeze ResNet backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Create ViTPose model
        self.vitpose = ViT(**vitpose_cfg)
        if pretrained_vitpose:
            self.vitpose.init_weights(pretrained_vitpose)
        
        # Adapter network to convert ResNet features to ViTPose input format
        self.adapter = nn.Sequential(
            # Initial channel reduction (2048 -> 512)
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            
            # Upsample to ViTPose input size (256x192) early
            nn.Upsample(size=self.target_size, mode='bilinear', align_corners=True),
            
            # Progressive channel reduction with spatial processing at target resolution
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            
            # Final adaptation to match ViTPose input (3 channels)
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            
            nn.Conv2d(64, 3, kernel_size=1),
            nn.BatchNorm2d(3),
        )
        
        # Save input sizes for reference
        self.square_input_size = square_input_size
        self.target_size = vitpose_cfg.get('img_size', (256, 192))
        
        # Initialize adapter weights
        self._init_adapter_weights()
    
    def _init_adapter_weights(self):
        """Initialize the weights of the adapter network."""
        for m in self.adapter.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward function.
        
        Args:
            x (Tensor): Input tensor, expected to be [B, 3, H, W]
                where H and W match self.square_input_size
        
        Returns:
            Tensor: Output tensor from ViTPose
        """
        # Resize input to square for ResNet
        B, C, H, W = x.shape
        if H != self.square_input_size or W != self.square_input_size:
            x = nn.functional.interpolate(
                x,
                size=(self.square_input_size, self.square_input_size),
                mode='bilinear',
                align_corners=True
            )
        
        # Extract features with ResNet
        x = self.backbone(x)
        
        # Adapt features
        x = self.adapter(x)
        
        # Normalize features
        x = x - x.mean(dim=(2, 3), keepdim=True)
        x = x / (x.std(dim=(2, 3), keepdim=True) + 1e-6)
        
        # No need to resize here since adapter already outputs correct size
        
        # Pass through ViTPose
        x = self.vitpose(x)
        
        return x
    
    def train(self, mode=True):
        """Convert the model into training mode while keeping some layers frozen."""
        super(ResNetViTPose, self).train(mode)
        if mode and self.backbone[0].training:
            # Keep ResNet in eval mode if frozen
            self.backbone.eval()
        return self