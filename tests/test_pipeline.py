import pytest
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from modeling.diffusion import DiffusionModel
from modeling.training import train_step, train_epoch, generate_samples
from modeling.unet import UnetModel


@pytest.fixture()
def fix_seed():
    torch.manual_seed(17)


@pytest.fixture
def train_dataset():
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = CIFAR10(
        "./data",
        train=True,
        download=True,
        transform=tf,
    )
    return dataset


@pytest.mark.parametrize(["device"], [["cpu"], ["cuda"]])
def test_train_on_one_batch(device, train_dataset):
    # note: you should not need to increase the threshold or change the hyperparameters
    ddpm = DiffusionModel(
        eps_model=UnetModel(3, 3, hidden_size=32),
        betas=(1e-4, 0.02),
        num_timesteps=1000,
    )
    ddpm.to(device)

    optim = torch.optim.Adam(ddpm.parameters(), lr=5e-4)
    dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    x, _ = next(iter(dataloader))
    loss = None
    for i in range(50):
        loss = train_step(ddpm, x, optim, device)
    assert loss < 0.5


@pytest.mark.parametrize(
    [
        'device',
        'betas',
        'num_timestamps',
        'eps_model_hidden',
    ],
    [
        (
            'cuda',
            (1e-4, 0.02),
            100,
            32
        ),
        (
            'cpu',
            (5e-4, 0.01),
            100,
            8
        ),
    ]
)
def test_training(fix_seed, tmp_path, train_dataset, device, betas, num_timestamps, eps_model_hidden):
    # note: implement and test a complete training procedure (including sampling)
    ddpm = DiffusionModel(
        eps_model=UnetModel(3, 3, hidden_size=eps_model_hidden),
        betas=betas,
        num_timesteps=num_timestamps,
    )
    ddpm.to(device)
    optim = torch.optim.Adam(ddpm.parameters(), lr=1e-4)

    dataloader = DataLoader(torch.utils.data.Subset(train_dataset, list(range(0, 256))), batch_size=128, shuffle=True)

    losses = []
    for i in range(2):
        losses.append(train_epoch(ddpm, dataloader, optim, device).data.to('cpu').numpy())

    assert losses[0] >= losses[1]
    assert losses[1] > 0
    assert losses[0] <= 1.3

    samples = ddpm.sample(2, (3, 32, 32), device)
    assert samples.shape == (2, 3, 32, 32)

    generate_samples(ddpm, device, str(tmp_path / 'test.jpg'))
