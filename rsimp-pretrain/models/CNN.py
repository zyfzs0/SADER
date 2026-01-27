import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple CNN for image completion.
    Expects inputs: x [B, H, W, C], mask [B, H, W, 1]
    """

    def __init__(self, configs):
        super().__init__()
        self.img_size = configs.image_size
        self.in_chans = configs.c_in
        self.hidden_dim = configs.d_model
        self.dropout_prob = configs.dropout

        self.block1 = nn.Sequential(
            nn.Conv2d(self.in_chans, self.hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_prob)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_prob)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_prob)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_prob)
        )
        self.final_conv = nn.Conv2d(self.hidden_dim, self.in_chans, kernel_size=3, padding=1)

    def forward(self, x, mask):
        """
        x: [B, H, W, C]
        mask: [B, H, W, 1], 1: visible, 0: missing
        """
        x = x * mask  # Apply mask to input
        original_x = x.clone()
        x = x.permute(0, 3, 1, 2)
        mask = mask.permute(0, 3, 1, 2)
        x_masked = x * mask

        out1 = self.block1(x_masked)
        out2 = self.block2(out1) + out1
        out3 = self.block3(out2) + out2
        out4 = self.block4(out3) + out3

        out = self.final_conv(out4) + x_masked
        out = out.permute(0, 2, 3, 1)

         # --- CRITICAL CONSTRAINT: Preserve visible regions ---
        # Create spatial mask for constraint enforcement
        spatial_mask = mask[:, :, :, 0:1]  # Use first channel [B, H, W, 1]
        
        # Enforce constraint: output = (prediction * masked_regions) + (original * visible_regions)
        final_output = out * (1 - spatial_mask) + original_x * spatial_mask

        return final_output
