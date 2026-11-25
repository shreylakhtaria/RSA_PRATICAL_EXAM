import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cpu")

transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)


class ChequeDigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def train_baseline(model, loader, epochs=2, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


def evaluate_clean(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def fgsm_attack(model, images, labels, epsilon):
    if epsilon == 0.0:
        return images.clone().detach()
    adv_images = images.clone().detach().requires_grad_(True)
    outputs = model(adv_images)
    loss = F.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()
    perturbed = adv_images + epsilon * adv_images.grad.sign()
    return torch.clamp(perturbed, 0.0, 1.0).detach()


def pgd_attack(model, images, labels, epsilon, steps=10):
    if epsilon == 0.0:
        return images.clone().detach()
    alpha = epsilon / 4.0
    adv_images = images.clone().detach()
    adv_images = adv_images + torch.empty_like(adv_images).uniform_(-epsilon, epsilon)
    adv_images = torch.clamp(adv_images, 0.0, 1.0).detach()
    for _ in range(steps):
        adv_images.requires_grad_(True)
        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            adv_images = adv_images + alpha * adv_images.grad.sign()
            delta = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)
            adv_images = torch.clamp(images + delta, 0.0, 1.0).detach()
    return adv_images


def evaluate_attack(model, loader, attack_fn, epsilon):
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        adv_images = attack_fn(model, images, labels, epsilon)
        with torch.no_grad():
            preds = model(adv_images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def save_clean_vs_adversarial(model, loader, epsilon=0.20, filename="clean_vs_adversarial.png"):
    model.eval()
    images, labels = next(iter(loader))
    images, labels = images[:5].to(device), labels[:5].to(device)
    with torch.no_grad():
        clean_preds = model(images).argmax(dim=1)
    adv_images = fgsm_attack(model, images, labels, epsilon)
    with torch.no_grad():
        adv_preds = model(adv_images).argmax(dim=1)

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    misread = []
    for i in range(5):
        axes[0, i].imshow(images[i].squeeze().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"True: {labels[i].item()} | Pred: {clean_preds[i].item()}")
        axes[0, i].axis("off")

        axes[1, i].imshow(adv_images[i].squeeze().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        color = "red" if adv_preds[i].item() != labels[i].item() else "green"
        axes[1, i].set_title(f"FGSM Pred: {adv_preds[i].item()}", color=color)
        axes[1, i].axis("off")

        if adv_preds[i].item() != labels[i].item():
            misread.append((i + 1, labels[i].item(), adv_preds[i].item()))

    axes[0, 0].set_ylabel("Clean", fontsize=12)
    axes[1, 0].set_ylabel(f"FGSM (eps={epsilon:.2f})", fontsize=12)
    plt.suptitle(f"Clean vs FGSM Adversarial Digits (epsilon = {epsilon:.2f})", fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Mis-read digits saved : {filename}")
    for idx, true_lbl, pred_lbl in misread:
        print(f"Sample {idx}: True digit {true_lbl} mis-read as {pred_lbl}")


def adversarial_train(model, loader, epochs=2, lr=0.001, epsilon=0.20):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            half = images.size(0) // 2
            clean_x, clean_y = images[:half], labels[:half]
            adv_src_x, adv_src_y = images[half:], labels[half:]

            model.eval()
            adv_x = fgsm_attack(model, adv_src_x, adv_src_y, epsilon)
            model.train()

            mixed_x = torch.cat([clean_x, adv_x], dim=0)
            mixed_y = torch.cat([clean_y, adv_src_y], dim=0)

            optimizer.zero_grad()
            outputs = model(mixed_x)
            loss = criterion(outputs, mixed_y)
            loss.backward()
            optimizer.step()


def plot_accuracy_vs_epsilon(epsilons, fgsm_accs, pgd_accs, filename="accuracy_vs_epsilon.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(epsilons, fgsm_accs, marker="o", linewidth=2, label="FGSM")
    plt.plot(epsilons, pgd_accs, marker="s", linewidth=2, label="PGD (10 steps)")
    plt.axhline(50.0, color="gray", linestyle="--", linewidth=1, label="50% Threshold")
    plt.title("Baseline CNN Accuracy vs Perturbation Budget (Epsilon)")
    plt.xlabel("Epsilon (L-infinity perturbation budget)")
    plt.ylabel("Test Accuracy (%)")
    plt.ylim(0, 105)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Plot saved : {filename}")


def main():
    baseline_model = ChequeDigitCNN().to(device)
    print("Training baseline CNN (2 epochs)...", end=" ", flush=True)
    train_baseline(baseline_model, train_loader, epochs=2, lr=0.001)
    print("done.")

    clean_acc = evaluate_clean(baseline_model, test_loader)
    print(f"Clean test accuracy : {clean_acc:.1f} %\n")

    epsilons = [0.00, 0.05, 0.10, 0.20, 0.30]
    fgsm_accs = []
    pgd_accs = []

    print(f"{'EPSILON':<10}{'FGSM_ACC':<12}{'PGD_ACC':<12}")
    print("-" * 31)
    for eps in epsilons:
        if eps == 0.0:
            f_acc = clean_acc
            p_acc = clean_acc
        else:
            f_acc = evaluate_attack(baseline_model, test_loader, fgsm_attack, eps)
            p_acc = evaluate_attack(baseline_model, test_loader, pgd_attack, eps)
        fgsm_accs.append(f_acc)
        pgd_accs.append(p_acc)
        print(f" {eps:<9.2f}{f_acc:>6.1f} %    {p_acc:>6.1f} %")
    print("-" * 31)

    save_clean_vs_adversarial(baseline_model, test_loader, epsilon=0.20)
    idx_020 = epsilons.index(0.20)
    fgsm_sr = 100.0 - fgsm_accs[idx_020]
    pgd_sr = 100.0 - pgd_accs[idx_020]
    print(f"Attack success (eps=0.20): FGSM {fgsm_sr:.1f} %, PGD {pgd_sr:.1f} %\n")

    adv_model = copy.deepcopy(baseline_model)
    print("Adversarial training (2 epochs, 50:50 clean + FGSM eps=0.20)...", end=" ", flush=True)
    adversarial_train(adv_model, train_loader, epochs=2, lr=0.001, epsilon=0.20)
    print("done.\n")

    adv_clean_acc = evaluate_clean(adv_model, test_loader)
    adv_fgsm_acc = evaluate_attack(adv_model, test_loader, fgsm_attack, 0.20)
    adv_pgd_acc = evaluate_attack(adv_model, test_loader, pgd_attack, 0.20)

    print(f"{'MODEL':<18}{'CLEAN':<10}{'FGSM(0.20)':<13}{'PGD(0.20)':<10}")
    print("-" * 52)
    print(f"{'Baseline':<18}{clean_acc:>5.1f} %   {fgsm_accs[idx_020]:>6.1f} %     {pgd_accs[idx_020]:>5.1f} %")
    print(f"{'Adversarially':<18}{adv_clean_acc:>5.1f} %   {adv_fgsm_acc:>6.1f} %     {adv_pgd_acc:>5.1f} %")
    print(f"{'trained':<18}")
    print("-" * 52)
    print(f"Clean accuracy change : {adv_clean_acc - clean_acc:+.1f} %")
    print(f"FGSM robustness gain  : {adv_fgsm_acc - fgsm_accs[idx_020]:+.1f} %   PGD gain : {adv_pgd_acc - pgd_accs[idx_020]:+.1f} %")

    plot_accuracy_vs_epsilon(epsilons, fgsm_accs, pgd_accs)


if __name__ == "__main__":
    main()
