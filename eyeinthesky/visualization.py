from PIL import Image
import io
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
import wandb

class FeatureVisualizationCallback:
    """
    Callback for visualizing feature maps during distillation training.
    
    Periodically captures and visualizes feature activations from specified layers
    of the student model to provide insights into the distillation process. The
    visualizations are logged to Weights & Biases as images.
    
    The callback works by:
    1. Registering hooks to capture feature maps at the start of specified epochs
    2. Creating grid visualizations of feature channels
    3. Logging these visualizations to W&B for monitoring and analysis
    
    Args:
        layers (list): Layer indices to visualize features from (from config)
        interval (int): Epoch interval between visualizations (from config)
        figsize (tuple): Size of the generated matplotlib figures
        grid_size (int): Number of feature channels per row/column in the grid
    
    Visualization only occurs every `interval` epochs to avoid excessive overhead
    while still providing useful insights into the feature learning process.
    """
    
    def __init__(self, layers, interval=20, figsize=(8.0, 8.0), grid_size=2):
        self.layers = layers
        self.interval = interval
        self.features = {}
        self.hooks = {}
        self.figsize = figsize
        self.grid_size = grid_size  # Number of images per row and column in each grid
        
    def set_hooks(self, trainer):
        # Clear previous features and hooks
        self.features = {}
        
        # Only register hooks on visualization epochs
        if (trainer.epoch) % self.interval != 0:
            return
            
        # Hook function to capture features
        def get_features(layer_idx):
            def hook(module, input, output):
                # Store just one image from batch for visualization
                self.features[layer_idx] = output[0].detach().clone()
            return hook
        
        # Register hooks for both models
        for layer_idx in self.layers:
            self.hooks[layer_idx] = trainer.model.model[layer_idx].register_forward_hook(
                get_features(layer_idx)
            )
    
    def plot_figures(self, trainer):
        # Skip if not visualization epoch
        if (trainer.epoch) % self.interval != 0:
            return
            
        # Remove hooks
        for hook in self.hooks.values():
            hook.remove()
   
        # Create visualizations for each layer
        for layer_idx, feature_maps in self.features.items():
            # Select all channels for visualization
            num_channels = feature_maps.shape[0]
            feature_subset = feature_maps.cpu()
            
            # Normalize each channel for better visualization
            for i in range(feature_subset.size(0)):
                min_val = feature_subset[i].min()
                max_val = feature_subset[i].max()
                if max_val > min_val:
                    feature_subset[i] = (feature_subset[i] - min_val) / (max_val - min_val)
            
            # Calculate how many grids we need to display all channels
            channels_per_grid = self.grid_size * self.grid_size
            num_grids = (num_channels + channels_per_grid - 1) // channels_per_grid  # Ceiling division
            
            # Create and log multiple grid images
            for grid_idx in range(num_grids):
                start_idx = grid_idx * channels_per_grid
                end_idx = min(start_idx + channels_per_grid, num_channels)
                current_channels = feature_subset[start_idx:end_idx]
                
                # Skip if no channels in this grid
                if len(current_channels) == 0:
                    continue
                
                # Create grid of feature maps
                grid = vutils.make_grid(
                    current_channels.unsqueeze(1),
                    nrow=self.grid_size,
                    padding=2
                )
                
                # Convert grid to numpy for matplotlib
                grid_np = grid.permute(1, 2, 0).numpy().astype(np.float32)
                
                # Create a matplotlib figure with specified size
                plt.figure(figsize=self.figsize)
                plt.imshow(grid_np)
                plt.title(f"Layer {layer_idx} Features - Epoch {trainer.epoch+1} - Group {grid_idx+1}/{num_grids}")
                plt.axis('off')
                plt.tight_layout()
                
                # Save figure to a buffer
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100)
                plt.close()
                buf.seek(0)
                
                # Convert buffer to PIL Image
                feature_image = Image.open(buf)
                
                # Log to wandb
                wandb.log({f"features/layer_{layer_idx}_epoch_{trainer.epoch+1}_group_{grid_idx+1}": wandb.Image(feature_image)}, commit=False)