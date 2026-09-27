cat << 'EOF' > README.md
# Generative Adversarial Networks: Face Synthesis & 1D Adversarial Dynamics

PyTorch implementation and empirical analysis of Generative Adversarial Networks (GANs), Conditional GANs (c-GANs) for continuous attribute control, and 1D adversarial probability density matching.

---

## Project Structure

```text
├── homework3_faces_gan.py    # Unconditional & Conditional (c-GAN) face generation
├── homework3_elbo_1d_gan.py   # 1D GAN simulation & theoretical density analysis
├── outputs_gan/              # Output face collages & loss curves (Part 1)
├── outputs_elbo/             # 1D empirical density & discriminator curves (Part 2)
├── .gitignore
└── README.md