import torch.nn as nn

class FeatureAdaptation(nn.Module):
    """
    Custom neural network layer that adapts student features to match teacher dimensions.
    
    This module performs feature transformation between different network architectures,
    enabling more effective knowledge distillation. It includes:
    
    1. Channel attention gating to emphasize important features
    2. Spatial context modules (with varying complexity based on layer importance)
    3. Channel dimension adaptation
    
    The adaptation process allows for comparing features from different network sizes
    (e.g., YOLOv12n student vs. YOLOv12x teacher) by transforming the student's
    feature space to align with the teacher's.
    
    Args:
        in_channels (int): Number of input channels (from student model)
        out_channels (int): Number of output channels (to match teacher model)
        layer_idx (int, optional): Layer index to determine adaptation complexity
    """
    
    def __init__(self, in_channels, out_channels, layer_idx=None):
        super().__init__()
        
        # Determine if this layer should use spatial context
        use_spatial = layer_idx in [8, 11, 14, 17, 20] 
        
        # Add gating mechanism for all layers
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global pooling to get channel-wise statistics
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(inplace=True),    # ReLU for non-linearity
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()              # Output in range [0,1] to act as gates
        )
        
        if use_spatial:
            # For the most critical layers, use full depthwise
            self.spatial = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
           
        else:
            # For moderately important layers, use simpler spatial context
            self.spatial = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels//4)
            
        # Regular channel adapter
        self.adapter = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
        # Initialize with Kaiming normalization
        nn.init.kaiming_normal_(self.adapter.weight)

        # Initialize with small weights to maintain stability
        nn.init.xavier_normal_(self.spatial.weight, gain=0.1)
            
        # Initialize gate with small weights to start with mild gating
        for m in self.gate.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Compute importance gates
        importance = self.gate(x)
        
        # Applies attention weights to features to emphatize the most informative features
        gated_features = x * importance
        
        # Apply spatial context if available
        if self.spatial is not None:
            # Apply spatial context with residual connection (on gated features)
            gated_features = self.spatial(gated_features) + gated_features
            
        # Apply channel adaptation
        return self.adapter(gated_features)