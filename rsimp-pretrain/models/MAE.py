import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.utils.rnn import pad_sequence

class PatchEmbedding(nn.Module):
    """Image to Patch Embedding"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)

class PositionalEncoding(nn.Module):
    """Learnable positional encoding"""
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        return x + self.pos_embed

class TransformerBlock(nn.Module):
    """Transformer block with pre-normalization"""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_output)
        
        x_norm = self.norm2(x)
        mlp_output = self.mlp(x_norm)
        x = x + self.dropout(mlp_output)
        
        return x

class TransformerEncoder(nn.Module):
    """Stack of Transformer blocks for encoder"""
    def __init__(self, embed_dim, num_heads, num_layers, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class TransformerDecoder(nn.Module):
    """Stack of Transformer blocks for decoder"""
    def __init__(self, embed_dim, num_heads, num_layers, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class Model(nn.Module):
    """
    Simple Masked Autoencoder (MAE) for image completion.
    Expects inputs: x [B, H, W, C], mask [B, H, W, 1]
    
    True MAE implementation with proper masking strategy for inpainting
    """
    def __init__(self, configs):
        super().__init__()
        self.img_size = configs.image_size
        self.in_chans = configs.c_in
        self.embed_dim = configs.d_model
        self.dropout_prob = configs.dropout

        # MAE-specific parameters with defaults
        self.patch_size = configs.patch_size
        self.num_heads = configs.num_heads
        self.encoder_layers = configs.encoder_layers
        self.decoder_layers = configs.decoder_layers
        self.mlp_ratio = configs.mlp_ratio
        self.mask_ratio = configs.mask_ratio
        
        self.num_patches = (self.img_size // self.patch_size) ** 2
        
        self.patch_embed = PatchEmbedding(
            self.img_size, self.patch_size, self.in_chans, self.embed_dim
        )
        
        self.pos_embed = PositionalEncoding(self.num_patches, self.embed_dim)
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        self.encoder = TransformerEncoder(
            self.embed_dim, self.num_heads, self.encoder_layers, 
            self.mlp_ratio, self.dropout_prob
        )
        
        self.decoder = TransformerDecoder(
            self.embed_dim, self.num_heads, self.decoder_layers,
            self.mlp_ratio, self.dropout_prob
        )
        
        self.head = nn.Linear(self.embed_dim, self.patch_size ** 2 * self.in_chans)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _spatial_mask_to_patch_mask(self, spatial_mask):
        """Convert spatial mask to patch-level mask"""
        mask = spatial_mask.permute(0, 3, 1, 2)[:, 0:1, :, :]
        mask = F.avg_pool2d(mask.float(), kernel_size=self.patch_size, stride=self.patch_size)
        mask = (mask > 0.5).squeeze(1)
        return mask.flatten(1)

    def _create_inpainting_mask(self, original_mask, additional_mask_ratio=0.0):
        """
        Create masking strategy for inpainting task
        original_mask: [B, num_patches] - 1: user-masked (to be inpainted), 0: visible
        additional_mask_ratio: ratio of visible patches to additionally mask for training
        """
        batch_size, num_patches = original_mask.shape
        
        if additional_mask_ratio > 0 and self.training:
            final_mask = original_mask.clone()
            
            for i in range(batch_size):
                visible_indices = torch.where(original_mask[i] == 0)[0]
                num_additional = int(len(visible_indices) * additional_mask_ratio)
                
                if num_additional > 0:
                    additional_mask = torch.randperm(len(visible_indices))[:num_additional]
                    final_mask[i, visible_indices[additional_mask]] = 1
                    
            return final_mask
        else:
            return original_mask

    def _reconstruct_image_from_patches(self, patch_pred, batch_size):
        """Reconstruct image from patch predictions"""
        patch_pred = patch_pred.view(batch_size, self.num_patches, self.in_chans, 
                                   self.patch_size, self.patch_size)
        patch_pred = patch_pred.permute(0, 2, 1, 3, 4)
        patch_pred = patch_pred.contiguous().view(batch_size, self.in_chans, 
                                                self.img_size, self.img_size)
        return patch_pred.permute(0, 2, 3, 1)

    def forward(self, x, mask):
        """
        x: [B, H, W, C]
        mask: [B, H, W, 1], 1: visible, 0: missing (to be inpainted)
        
        Key constraint: Visible regions in input must remain unchanged in output
        """
        mask = mask.float()
        batch_size = x.shape[0]
        original_x = x.clone()  # Keep original for constraint enforcement
        
        # Convert to channel-first format
        x_tensor = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # Convert spatial mask to patch-level mask
        # Invert mask: 1->masked (to inpaint), 0->visible
        patch_mask = torch.logical_not(self._spatial_mask_to_patch_mask(mask)).float()
        
        # Apply additional masking during training
        inpainting_mask = self._create_inpainting_mask(patch_mask, additional_mask_ratio=0.25)
        
        # Patch embedding
        x_patch = self.patch_embed(x_tensor)  # [B, num_patches, embed_dim]
        x_patch = self.pos_embed(x_patch)
        
        # --- ENCODER: Process only non-masked patches ---
        visible_indices = [torch.where(inpainting_mask[i] == 0)[0] for i in range(batch_size)]
        masked_indices = [torch.where(inpainting_mask[i] == 1)[0] for i in range(batch_size)]
        
        visible_list = [x_patch[i, visible_indices[i]] for i in range(batch_size)]
        x_visible = pad_sequence(visible_list, batch_first=True)  # [B, max_visible, embed_dim]
        
        latent = self.encoder(x_visible)
        
        # --- DECODER: Process all patches ---
        decoder_input = torch.zeros(batch_size, self.num_patches, self.embed_dim, 
                                    device=x.device, dtype=x_patch.dtype)
        
        # prepare a 2D mask token once (shape: [1, embed_dim])
        mask_token_2d = self.mask_token.view(1, self.embed_dim) if self.mask_token.dim() == 3 else self.mask_token
        # ensure it's on the correct device/dtype
        mask_token_2d = mask_token_2d.to(device=decoder_input.device, dtype=decoder_input.dtype)

        for i in range(batch_size):
            valid_len = visible_list[i].shape[0]
            # latent may have padding; only take valid_len tokens
            decoder_input[i, visible_indices[i]] = latent[i, :valid_len]
            num_masked = masked_indices[i].shape[0]
            if num_masked > 0:
                decoder_input[i, masked_indices[i]] = mask_token_2d.expand(num_masked, self.embed_dim)
        
        decoder_input = self.pos_embed(decoder_input)
        decoded = self.decoder(decoder_input)
        pred_pixels = self.head(decoded)
        
        # Reconstruct image from patches
        recon = self._reconstruct_image_from_patches(pred_pixels, batch_size)
        
        # --- CRITICAL CONSTRAINT: Preserve visible regions ---
        # Create spatial mask for constraint enforcement
        spatial_mask = mask[:, :, :, 0:1]  # Use first channel [B, H, W, 1]
        
        # Enforce constraint: output = (prediction * masked_regions) + (original * visible_regions)
        final_output = recon * (1 - spatial_mask) + original_x * spatial_mask
        
        return final_output