import os
import matplotlib.pyplot as plt
import torch
import numpy as np

def visualize_images(original, reconstructed, titles=None, save_path=None, s2_rgb_indices=(3, 2, 1)):
    """
    Visualize original and reconstructed images side by side, handling different channel cases:
    - 3 channels: RGB
    - 4 channels: RGB + NIR
    - 13 channels: Sentinel-2 (default use B4, B3, B2 as RGB)
    
    Args:
        original, reconstructed: torch.Tensor or np.ndarray, shape (B, H, W, C) or (B, C, H, W)
        titles: optional list of titles
        save_path: path to save figure
        s2_rgb_indices: tuple, which 3 Sentinel-2 channels to use for RGB visualization (default: (3,2,1)=B4,B3,B2)
    """
    def to_hwc(x):
        """Convert to (B, H, W, C) numpy array."""
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if x.ndim == 4 and x.shape[1] <= 13:  # (B, C, H, W) -> (B, H, W, C)
            x = x.transpose(0, 2, 3, 1)
        return x

    def normalize_uint8(img):
        """Normalize to [0,255] uint8."""
        img = np.clip(img * 255.0, 0, 255)
        return img.astype('uint8')

    orig_imgs = to_hwc(original)
    rec_imgs  = to_hwc(reconstructed)
    num_images = orig_imgs.shape[0]
    channels   = orig_imgs.shape[-1]

    # Figure layout
    plt.figure(figsize=(5 * num_images, 8))

    for i in range(num_images):
        if channels == 3:
            # RGB
            plt.subplot(2, num_images, i + 1)
            plt.imshow(normalize_uint8(orig_imgs[i]))
            plt.axis('off')
            if titles:
                plt.title(f"{titles[i]} - Orig (RGB)")

            plt.subplot(2, num_images, i + 1 + num_images)
            plt.imshow(normalize_uint8(rec_imgs[i]))
            plt.axis('off')
            if titles:
                plt.title(f"{titles[i]} - Rec (RGB)")

        elif channels == 4:
            # RGB
            plt.subplot(3, num_images, i + 1)
            plt.imshow(normalize_uint8(orig_imgs[i, :, :, :3]))
            plt.axis('off')
            if titles:
                plt.title(f"{titles[i]} - Orig (RGB)")

            plt.subplot(3, num_images, i + 1 + num_images)
            plt.imshow(normalize_uint8(rec_imgs[i, :, :, :3]))
            plt.axis('off')
            if titles:
                plt.title(f"{titles[i]} - Rec (RGB)")

            # NIR
            plt.subplot(3, num_images, i + 1 + 2 * num_images)
            nir_orig = orig_imgs[i, :, :, 3]
            nir_rec  = rec_imgs[i,  :, :, 3]
            nir_merge = np.concatenate([nir_orig, nir_rec], axis=1)
            plt.imshow(normalize_uint8(nir_merge), cmap='gray')
            plt.axis('off')
            if titles:
                plt.title(f"{titles[i]} - NIR (Orig|Rec)")

        elif channels == 13:
            # use np.take to avoid advanced-indexing axis moves
            rgb_orig = np.take(orig_imgs[i], s2_rgb_indices, axis=-1)
            rgb_rec  = np.take(rec_imgs[i],  s2_rgb_indices, axis=-1)

            plt.subplot(2, num_images, i + 1)
            plt.imshow(normalize_uint8(rgb_orig))
            plt.axis('off')
            if titles:
                plt.title(f"{titles[i]} - Orig (S2 RGB)")

            plt.subplot(2, num_images, i + 1 + num_images)
            plt.imshow(normalize_uint8(rgb_rec))
            plt.axis('off')
            if titles:
                plt.title(f"{titles[i]} - Rec (S2 RGB)")

        else:
            raise ValueError(f"Unsupported number of channels: {channels}")

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()
