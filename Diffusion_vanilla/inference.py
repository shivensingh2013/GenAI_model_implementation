from config import CONFIG
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
from diffusion_reverse import DiffusionReverseProcess
import numpy as np
import pandas as pd


def generate(cfg):

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Initialize Diffusion Reverse Process
    drp = DiffusionReverseProcess()
    # Set model to eval mode
    model = torch.load(cfg.model_path).to(device)
    model.eval()

    # Generate Noise sample from N(0, 1)
    xt = torch.randn(1, cfg.in_channels, cfg.img_size, cfg.img_size).to(device)

     # Denoise step by step by going backward.
    with torch.no_grad():
        for t in reversed(range(cfg.num_timesteps)):
            noise_pred = model(xt, torch.as_tensor(t).unsqueeze(0).to(device))
            xt, x0 = drp.sample_prev_timestep(xt, noise_pred, torch.as_tensor(t).to(device))
    # Convert the image to proper scale
    xt = torch.clamp(xt, -1., 1.).detach().cpu()
    xt = (xt + 1) / 2

    return xt



if __name__ == "__main__":
    cfg = CONFIG()

    generated_imgs = []

    for i in tqdm(range(cfg.num_img_to_generate)):
        xt = generate(cfg)
        xt = 255 * xt[0][0].numpy()
        generated_imgs.append(xt.astype(np.uint8).flatten())

# Save Generated Data CSV
    generated_df = pd.DataFrame(generated_imgs, columns=[f'pixel{i}' for i in range(784)])
    generated_df.to_csv(cfg.generated_csv_path, index=False)

# Visualize
    fig, axes = plt.subplots(4, 1, figsize=(8, 8))

# Plot each image in the corresponding subplot
    for i, ax in enumerate(axes.flat):
        ax.imshow(np.reshape(generated_imgs[i], (28, 28)), cmap='gray')  # You might need to adjust the colormap based on your images
        ax.axis('off')  # Turn off axis labels

    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()

