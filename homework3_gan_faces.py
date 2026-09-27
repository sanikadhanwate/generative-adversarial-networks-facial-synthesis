import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

IMAGE_SIZE = 24
INPUT_DIM = IMAGE_SIZE * IMAGE_SIZE
LATENT_DIM = 64
BATCH_SIZE = 128
AGE_SCALE = 100.0


class Generator(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, conditional=False):
        super().__init__()
        self.conditional = conditional
        in_dim = latent_dim + 1 if conditional else latent_dim
        self.network = nn.Sequential(
            nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, INPUT_DIM), nn.Sigmoid())

    def forward(self, z, y=None):
        if self.conditional:
            z = torch.cat([z, y], dim=1)
        return self.network(z)


class Discriminator(nn.Module):
    def __init__(self, conditional=False):
        super().__init__()
        self.conditional = conditional
        in_dim = INPUT_DIM + 1 if conditional else INPUT_DIM
        self.network = nn.Sequential(
            nn.Linear(in_dim, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1), nn.Sigmoid())

    def forward(self, x, y=None):
        if self.conditional:
            x = torch.cat([x, y], dim=1)
        return self.network(x)


def save_image_grid(images, path, rows, cols, title=None):
    images = images.detach().cpu().numpy().reshape(-1, IMAGE_SIZE, IMAGE_SIZE)
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows), squeeze=False)
    for ax, image in zip(axes.flat, images):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1); ax.axis("off")
    for ax in axes.flat[len(images):]: ax.axis("off")
    if title: fig.suptitle(title)
    fig.tight_layout(); fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)


def train_gan(device, G, D, loader, epochs, latent_dim, conditional, out_dir, model_tag):
    opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(.5, .999))
    opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(.5, .999))
    bce, history = nn.BCELoss(), []
    for epoch in range(1, epochs + 1):
        totals = np.zeros(2)
        for batch in loader:
            x_real = batch[0].to(device); y_real = batch[1].to(device) if conditional else None
            n = x_real.size(0)
            z = torch.randn(n, latent_dim, device=device)
            y_fake = torch.rand(n, 1, device=device) if conditional else None
            x_fake = G(z, y_fake)

            #Discriminator
            opt_d.zero_grad()
            x_all = torch.cat([x_real, x_fake.detach()], dim=0)
            labels_all = torch.cat([torch.full((n, 1), .9, device=device), torch.zeros(n, 1, device=device)], dim=0)
            y_all = torch.cat([y_real, y_fake], dim=0) if conditional else None
            d_loss = bce(D(x_all, y_all), labels_all)
            d_loss.backward(); opt_d.step()

            #Generator
            opt_g.zero_grad()
            g_loss = bce(D(x_fake, y_fake), torch.ones(n, 1, device=device))
            g_loss.backward(); opt_g.step()

            totals += np.array([d_loss.item(), g_loss.item()]) * n
        row = totals / len(loader.dataset); history.append(row)
        print(f"{model_tag} {epoch:3d}/{epochs}: D_loss={row[0]:.3f}, G_loss={row[1]:.3f}")
    torch.save(G.state_dict(), out_dir / f"{model_tag}_generator.pt")
    torch.save(D.state_dict(), out_dir / f"{model_tag}_discriminator.pt")
    return np.asarray(history)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../Gen AI HW2/faces_vae.npy")
    parser.add_argument("--ages", default="ages23k.npy")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--cgan-epochs", type=int, default=100)
    parser.add_argument("--output", default="outputs_gan")
    args = parser.parse_args(); torch.manual_seed(7); np.random.seed(7)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print("Using", device)

    x_all = torch.from_numpy(np.load(args.data).astype(np.float32).reshape(-1, INPUT_DIM) / 255.)

    #Part 1(a)
    loader = DataLoader(TensorDataset(x_all), batch_size=BATCH_SIZE, shuffle=True, pin_memory=device.type == "cuda")
    G, D = Generator(LATENT_DIM).to(device), Discriminator().to(device)
    history = train_gan(device, G, D, loader, args.epochs, LATENT_DIM, False, out, "gan")
    fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(history[:, 0], label="D loss"); ax.plot(history[:, 1], label="G loss")
    ax.legend(); ax.set(xlabel="epoch", ylabel="loss", title="Unconditional GAN training"); fig.tight_layout()
    fig.savefig(out / "part1a_training_losses.png", dpi=170); plt.close(fig)
    G.eval()
    with torch.no_grad():
        samples = G(torch.randn(20, LATENT_DIM, device=device))
    save_image_grid(samples, out / "part1a_gan_samples.png", rows=2, cols=10, title="GAN samples (20)")

    #Part 1(b)
    ages = np.load(args.ages).astype(np.float32)
    valid = (ages >= 0) & (ages <= 100)
    x_valid, y_valid = x_all[valid], torch.from_numpy(ages[valid] / AGE_SCALE).unsqueeze(1)
    print(f"Conditional GAN: using {valid.sum()} / {len(ages)} images with a valid age label")
    cloader = DataLoader(TensorDataset(x_valid, y_valid), batch_size=BATCH_SIZE, shuffle=True, pin_memory=device.type == "cuda")
    cG, cD = Generator(LATENT_DIM, conditional=True).to(device), Discriminator(conditional=True).to(device)
    chistory = train_gan(device, cG, cD, cloader, args.cgan_epochs, LATENT_DIM, True, out, "cgan")
    fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(chistory[:, 0], label="D loss"); ax.plot(chistory[:, 1], label="G loss")
    ax.legend(); ax.set(xlabel="epoch", ylabel="loss", title="Conditional GAN training"); fig.tight_layout()
    fig.savefig(out / "part1b_training_losses.png", dpi=170); plt.close(fig)

    cG.eval()
    target_ages = [8 + i * 10 for i in range(1, 7)]
    with torch.no_grad():
        rows = []
        for age in target_ages:
            z = torch.randn(10, LATENT_DIM, device=device)
            y = torch.full((10, 1), age / AGE_SCALE, device=device)
            rows.append(cG(z, y))
        grid = torch.cat(rows, dim=0)
    save_image_grid(grid, out / "part1b_cgan_samples.png", rows=6, cols=10,
                     title="Conditional GAN samples: rows = ages " + ", ".join(map(str, target_ages)))
    print("Finished; see", out)


if __name__ == "__main__":
    main()
