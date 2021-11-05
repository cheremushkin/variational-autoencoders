import typing as tp
import argparse

import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
from torch import nn
from torch import optim
from torchvision.utils import save_image

from dataset import FacesDataset
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from vae import VAE
from elbo import ELBO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', '--input-folder')
    parser.add_argument('-output', '--output-folder')
    parser.add_argument('-size', '--img-size', default=128, type=int)
    parser.add_argument('-z', '--z-size', default=100, type=int)
    parser.add_argument('-e', '--n_epochs', default=10, type=int)
    parser.add_argument('-lr', '--learning-rate', default=1e-4, type=float)
    parser.add_argument('-bs', '--batch-size', default=16, type=int)
    parser.add_argument('-d', '--device', default='cuda')

    return parser.parse_args()


def train_epoch(dl: DataLoader,
                model: nn.Module,
                optimizer: optim.Optimizer,
                criterion: nn.Module):
    model.train()
    running_loss = 0.0
    for i, xb in tqdm(enumerate(dl), total=len(dl)):
        xb = xb.to(args.device)
        xb_hat, mu, log_var = model(xb)
        loss = criterion(xb_hat, xb, mu, log_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(dl.dataset)
    return train_loss


def validate_epoch(n_epoch: int,
                   dl: DataLoader,
                   model: nn.Module,
                   criterion: nn.Module,
                   output_folder: Path):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, xb in tqdm(enumerate(dl), total=len(dl)):
            xb = xb.to(args.device)
            xb_hat, mu, log_var = model(xb)
            loss = criterion(xb_hat, xb, mu, log_var)
            running_loss += loss.item()

            if i == int(len(dl.dataset) / dl.batch_size) - 1:
                n_rows = 8
                both = torch.cat((xb[:n_rows],
                                  xb_hat[:n_rows]))
                save_image(both.cpu(), str(output_folder / f'imgs/{n_epoch}.png'), nrow=n_rows)
    val_loss = running_loss / len(dl.dataset)
    return val_loss


def load_data(image_size: int,
              input_folder: Path,
              batch_size: int) -> tp.Tuple[DataLoader, DataLoader]:
    transforms = A.Compose([
        A.Resize(image_size, image_size),
        #     A.HorizontalFlip(p=0.5),
        #     A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.485, 0.485), std=(0.229, 0.229, 0.229)),
        ToTensorV2(p=1)
    ])

    ds = FacesDataset(folder=str(input_folder), transforms=transforms)

    n_train = int(len(ds) * 0.8)
    n_valid = len(ds) - n_train
    ds_train, ds_valid = torch.utils.data.random_split(ds, [n_train, n_valid])

    dl_train = DataLoader(ds_train, batch_size=batch_size, num_workers=2, shuffle=True)
    dl_valid = DataLoader(ds_valid, batch_size=batch_size, num_workers=2, shuffle=False)

    return dl_train, dl_valid


def save_model(n_epoch: int,
               model: nn.Module,
               optimizer: optim.Optimizer,
               output_folder: Path):
    torch.save({
        'epoch': n_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, str(output_folder / f'models/{n_epoch}.pth'))


def run_training(args):
    dl_train, dl_valid = load_data(args.img_size, args.input_folder, args.batch_size)

    vae = VAE(img_size=args.img_size,
              n_channels=3,
              n_features=16,
              n_layers=np.log2(args.img_size // 4).astype(int),
              z_size=args.z_size).to(args.device)

    criterion = ELBO(nn.BCELoss(reduction='sum'))
    optimizer = optim.Adam(vae.parameters(), lr=args.learning_rate)

    train_loss = []
    val_loss = []
    for n_epoch in range(args.n_epochs):
        print(f'Epoch {n_epoch + 1} of {args.n_epochs}')
        train_epoch_loss = train_epoch(dl_train, vae, optimizer, criterion)
        val_epoch_loss = validate_epoch(n_epoch, dl_valid, vae, criterion, Path(args.output_folder))
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        print(f'Train Loss:\t\t{train_epoch_loss:.4f}')
        print(f'Validation Loss:\t{val_epoch_loss:.4f}')
        save_model(n_epoch, vae, optimizer, Path(args.output_folder))


if __name__ == '__main__':
    args = parse_args()
    run_training(args)
