import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from modeling.diffusion import DiffusionModel
from modeling.training import generate_samples, train_epoch
from modeling.unet import UnetModel

import wandb
import yaml


def main(device: str, config: dict):
    ddpm = DiffusionModel(
        eps_model=UnetModel(3, 3, hidden_size=config['eps_model_hidden']),
        betas=(config['beta_1'], config['beta_2']),
        num_timesteps=config['num_timestamps'],
    )
    ddpm.to(device)

    train_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    dataset = CIFAR10(
        "cifar10",
        train=True,
        download=False,
        transform=train_transforms
    )

    dataloader = DataLoader(dataset, batch_size=config['batch_size'], num_workers=4, shuffle=True)
    optim = torch.optim.Adam(ddpm.parameters(), lr=config['lr'])

    wandb.init(
        project="effdl-hw1",
        config=config,
        entity='ignat'
    )

    for i in range(config['num_epochs']):
        loss = train_epoch(ddpm, dataloader, optim, device)
        generate_samples(ddpm, device, f"samples/{i:02d}.png")
        wandb.log(
            {
                'train loss': loss,
                'lr': optim.param_groups[0]['lr'],
                'samples': wandb.Image(f"samples/{i:02d}.png")
            }
        )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open('hyperparam.yaml') as config_file:
        config = yaml.safe_load(config_file)
        main(device=device, config=config)
