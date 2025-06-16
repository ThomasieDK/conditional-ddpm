import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.utils import make_grid
from torchvision import transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os



class SpriteDataset(Dataset):
    """Sprite dataset class"""
    def __init__(self, root, transform, target_transform):
        self.images = np.load(os.path.join(root, "sprites_1788_16x16.npy"))
        self.labels = np.load(os.path.join(root, "sprite_labels_nc_1788_16x16.npy"))
        self.transform = transform
        self.target_transform = target_transform
    
    def __getitem__(self, idx):
        image = self.transform(self.images[idx])
        label = self.target_transform(self.labels[idx])
        return image, label

    def __len__(self):
        return len(self.images)


class PairedImageDataset(Dataset):
    """Dataset class for paired images.
    Expects two subdirectories ``stained`` and ``unstained`` under ``root``.
    Images with the same file name are assumed to be pairs.
    """

    def __init__(self, root, transform=None):
        self.stained_dir = os.path.join(root, "stained")
        self.unstained_dir = os.path.join(root, "unstained")

        self.stain_paths = sorted([os.path.join(self.stained_dir, f)
                                   for f in os.listdir(self.stained_dir)
                                   if not f.startswith('.')])
        self.unstain_paths = sorted([os.path.join(self.unstained_dir, f)
                                     for f in os.listdir(self.unstained_dir)
                                     if not f.startswith('.')])
        assert len(self.stain_paths) == len(self.unstain_paths), "Mismatched dataset sizes"

        self.transform = transform or T.Compose([
            T.ToTensor(),
            lambda x: 2 * x - 1
        ])

    def __len__(self):
        return len(self.stain_paths)

    def __getitem__(self, idx):
        stain = Image.open(self.stain_paths[idx]).convert('RGB')
        unstain = Image.open(self.unstain_paths[idx]).convert('RGB')

        stain = self.transform(stain)
        unstain = self.transform(unstain)
        # return pair as (unstained, stained)
        return unstain, stain

def generate_animation(intermediate_samples, t_steps, fname, n_images_per_row=8):
    """Generates animation and saves as a gif file for given intermediate samples"""
    intermediate_samples = [make_grid(x, scale_each=True, normalize=True, 
                                      nrow=n_images_per_row).permute(1, 2, 0).numpy() for x in intermediate_samples]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axis("off")
    img_plot = ax.imshow(intermediate_samples[0])
    
    def update(frame):
        img_plot.set_array(intermediate_samples[frame])
        ax.set_title(f"T = {t_steps[frame]}")
        fig.tight_layout()
        return img_plot
    
    ani = FuncAnimation(fig, update, frames=len(intermediate_samples), interval=200)
    ani.save(fname)


def get_custom_context(n_samples, n_classes, device):
    """Returns custom context in one-hot encoded form"""
    context = []
    for i in range(n_classes - 1):
        context.extend([i]*(n_samples//n_classes))
    context.extend([n_classes - 1]*(n_samples - len(context)))
    return torch.nn.functional.one_hot(torch.tensor(context), n_classes).float().to(device)
