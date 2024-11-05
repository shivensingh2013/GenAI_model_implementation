# Imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import Unet
from diffusion_forward import DiffusionForward
from diffusion_reverse import DiffusionReverseProcess
from dataset import diffusion_dataset
import numpy as np
from config import CONFIG


def train(cfg):
        # Dataset and Dataloader - training
        
        mnist_ds = diffusion_dataset(cfg.train_csv_path)
        mnist_dl = DataLoader(mnist_ds, cfg.batch_size, shuffle=True)
           # Device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Device: {device}\n')

        # Initiate Model
        model = Unet().to(device)
        # Initialize Optimizer and Loss Function
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        criterion = torch.nn.MSELoss()

        # Diffusion Forward Process to add noise
        dfp = DiffusionForward()
        best_eval_loss = float('inf')

        for epoch in range(cfg.num_epochs):
            losses = []
            model.train()
            for imgs in tqdm(mnist_dl):
                imgs = imgs.to(device)
                # Generate noise and timestamps
                noise = torch.randn_like(imgs).to(device)
                t = torch.randint(0, cfg.num_timesteps, (imgs.shape[0],)).to(device)

                # Add noise to the images using Forward Process
                noisy_imgs = dfp.add_noise(imgs, noise, t)
                # Avoid Gradient Accumulation
                optimizer.zero_grad()

                # Predict noise using U-net Model
                noise_pred = model(noisy_imgs, t)

                # Calculate Loss
                loss = criterion(noise_pred, noise)
                losses.append(loss.item())
                 # Backprop + Update model params
                loss.backward()
                optimizer.step()

        # Mean Loss
            mean_epoch_loss = np.mean(losses)
        
        # Display
            print('Epoch:{} | Loss : {:.4f}'.format(
                epoch + 1,
                mean_epoch_loss,
            ))
        
        # Save based on train-loss
            if mean_epoch_loss < best_eval_loss:
                best_eval_loss = mean_epoch_loss
                torch.save(model, cfg.model_path)

        return print("Training complete")

if __name__ == "__main__":
# Config
    cfg = CONFIG()

# TRAIN
    train(cfg)







