from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

N_DATA, EPOCHS, BATCH_SIZE, Z_DIM, N_SNAPSHOTS = 4000, 2000, 256, 1, 10


def sample_pdata(n):
    comp = (torch.rand(n) < 0.5).float()
    return comp * (-2 + 0.5 * torch.randn(n)) + (1 - comp) * (2 + 0.5 * torch.randn(n))


class Generator1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(Z_DIM, 64), nn.ReLU(),
                                  nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, z):
        return self.net(z)


class Discriminator1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 64), nn.LeakyReLU(0.2),
                                  nn.Linear(64, 64), nn.LeakyReLU(0.2),
                                  nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x):
        return self.net(x)


def plot_snapshot(G, D, x_real, epoch, out):
    with torch.no_grad():
        z = torch.randn(N_DATA, Z_DIM)
        x_fake = G(z).squeeze(1)
        grid = torch.linspace(-5, 5, 400).unsqueeze(1)
        d_curve = D(grid).squeeze(1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(x_real.numpy(), bins=60, range=(-5, 5), density=True, alpha=0.5, color="tab:blue", label="P_data(x): real")
    ax.hist(x_fake.numpy(), bins=60, range=(-5, 5), density=True, alpha=0.5, color="tab:red", label="P(G(z)): generated")
    ax.plot(grid.squeeze(1).numpy(), d_curve.numpy(), color="tab:orange", lw=2, label="D(x)")
    ax.set(xlabel="x", ylabel="density / D(x)", title=f"epoch {epoch}", ylim=(0, 1.05))
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(out / f"snapshot_epoch_{epoch:04d}.png", dpi=150); plt.close(fig)


def main():
    torch.manual_seed(0)
    out = Path("outputs_gan_1d"); out.mkdir(exist_ok=True)
    x_real_full = sample_pdata(N_DATA)

    G, D = Generator1D(), Discriminator1D()
    opt_g = torch.optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, .999))
    opt_d = torch.optim.Adam(D.parameters(), lr=1e-3, betas=(0.5, .999))
    bce = nn.BCELoss()

    snapshot_epochs = np.linspace(0, EPOCHS, N_SNAPSHOTS, dtype=int)
    steps_per_epoch = N_DATA // BATCH_SIZE

    for epoch in range(EPOCHS + 1):
        if epoch in snapshot_epochs:
            plot_snapshot(G, D, x_real_full, epoch, out)
        if epoch == EPOCHS:
            break
        for _ in range(steps_per_epoch):
            x_real = sample_pdata(BATCH_SIZE).unsqueeze(1)
            z = torch.randn(BATCH_SIZE, Z_DIM)
            x_fake = G(z)

            opt_d.zero_grad()
            x_all = torch.cat([x_real, x_fake.detach()], dim=0)
            labels_all = torch.cat([torch.full((BATCH_SIZE, 1), 0.9), torch.zeros(BATCH_SIZE, 1)], dim=0)
            d_loss = bce(D(x_all), labels_all)
            d_loss.backward(); opt_d.step()

            opt_g.zero_grad()
            g_loss = bce(D(x_fake), torch.ones(BATCH_SIZE, 1))
            g_loss.backward(); opt_g.step()

    print("Finished; snapshots in", out)


if __name__ == "__main__":
    main()
