import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

import config
from config import *
from torchvision.utils import save_image
# from model import Discriminator, Generator
from Dataset import CustomDataset
from torch.utils.data import DataLoader, Dataset
from util import *
from tqdm import tqdm
from Eval import *
import numpy as np
import matplotlib.pyplot as plt


def train(mri_disc, ct_disc, mri_gen, ct_gen, disc_opt, gen_opt, train_loader, val_loader, mse, d_scaler, g_scaler, L1,
          folder):
    PNSR = PeakSignalNoiseRatio().to(config.device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(config.device)
    loss = nn.MSELoss().to(config.device)
    loop = tqdm(train_loader, leave=True)
    G_losses = []
    D_losses = []
    Disc_loss = 0
    Gen_loss = 0
    for i, (mri, ct) in enumerate(loop):
        mri = mri.to(config.device)
        ct = ct.to(config.device)

        # Train Generator

        with torch.cuda.amp.autocast():
            fake_ct = ct_gen(mri)
            fake_mri = ct_gen(ct)
            D_CT_fake = ct_disc(fake_ct)
            D_mri_fake = mri_disc(fake_mri)
            G_CT_loss = mse(D_CT_fake, torch.ones_like(D_CT_fake))
            G_mri_loss = mse(D_mri_fake, torch.ones_like(D_mri_fake))

            # Cycle loss

            cycle_mri = mri_gen(fake_ct)
            cycle_ct = ct_gen(fake_mri)
            cycle_mri_loss = L1(mri, cycle_mri)
            cycle_ct_loss = L1(ct, cycle_ct)
            G_loss = 0 * G_mri_loss + 0 * G_CT_loss + cycle_ct_loss * config.LAMBDA_CYCLE + cycle_mri_loss * config.LAMBDA_CYCLE
        gen_opt.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(gen_opt)
        g_scaler.update()

        # Train Discriminator

        with torch.cuda.amp.autocast():
            # fake_ct = ct_gen(mri)
            D_CT_real = ct_disc(ct)
            D_CT_fake = ct_disc(fake_ct.detach())
            D_CT_real_loss = mse(D_CT_real, torch.ones_like(D_CT_real))
            D_CT_fake_loss = mse(D_CT_fake, torch.zeros_like(D_CT_fake))
            D_CT_loss = D_CT_real_loss + D_CT_fake_loss

            # fake_mri = mri_gen(ct)
            D_mri_real = mri_disc(mri)
            D_mri_fake = mri_disc(fake_mri.detach())
            D_mri_real_loss = mse(D_mri_real, torch.ones_like(D_mri_real))
            D_mri_fake_loss = mse(D_mri_fake, torch.zeros_like(D_mri_fake))
            D_mri_loss = D_mri_real_loss + D_mri_fake_loss

            D_loss = (D_CT_loss + D_mri_loss) / 2
        disc_opt.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(disc_opt)
        d_scaler.update()

        Disc_loss = D_loss
        Gen_loss = G_loss
        D_losses.append(D_loss.cpu().detach().numpy())
        G_losses.append(G_loss.cpu().detach().numpy())

        if i % 10 == 0:
            save_image(mri * 0.5 + 0.5, f'Data/saved_img/train/mri_{i}.png')
            save_image(fake_mri * 0.5 + 0.5, f"Data/saved_img/train/fake_mri_{i}.png")
            save_image(ct * 0.5 + 0.5, f"Data/saved_img/train/ct_{i}.png")
            save_image(fake_ct * 0.5 + 0.5, f"Data/saved_img/train/fake_ct_{i}.png")

    for i, (mri, ct) in enumerate(val_loader):
        ct = ct.to(config.device)
        mri = mri.to(config.device)
        ct_gen.eval()
        mri_gen.eval()
        with torch.no_grad():
            fake_ct = ct_gen(mri)
            fake_mr = mri_gen(ct)
            PNSR_V = PNSR(ct * 0.5 + 0.5, fake_ct * 0.5 + 0.5)
            ssimm = ssim(ct * 0.5 + 0.5, fake_ct * 0.5 + 0.5)
            mse = loss(ct * 0.5 + 0.5, fake_ct * 0.5 + 0.5).item()
            fake_ct = fake_ct * 0.5 + 0.5
            ct_err = torch.abs((ct * 0.5 + 0.5) - fake_ct)
            if i % 2 == 0:
                save_image(fake_ct, folder + f"/fake_ct{i}.png")
                save_image(mri * 0.5 + 0.5, folder + f"/mri{i}.png")
                save_image(ct * 0.5 + 0.5, folder + f"/ct{i}.png")
                save_image(fake_mr * 0.5 + 0.5, folder + f"/fake_mr{i}.png")
                save_image(ct_err, folder + f"/ct_er{i}.png")
        mri_gen.train()
        ct_gen.train()
    print('Disc Loss: {:.5f}  Gen Loss: {:.5f} PNSR:{:.5f}  SSIM:{:.5f} MSE:{:.5f}'.format(Disc_loss, Gen_loss, PNSR_V,
                                                                                           ssimm, mse))
    return np.array(D_losses).mean(), np.array(G_losses).mean()


dataset = CustomDataset(config.train_dir_MR, config.train_dir_CT, transform=config.transforms)
train_dataset, valid_dataset = train_test_split(dataset, test_size=0.1, random_state=42)
val_dataset, test_dataset = train_test_split(valid_dataset, test_size=0.5, random_state=42)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
valid_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


def main():
    mri_disc = Discriminator(in_channel=3, out_channel=1).to(config.device)
    ct_disc = Discriminator(in_channel=3, out_channel=1).to(config.device)
    mri_gen = Generator(in_channel=3, num_residual=9).to(config.device)
    ct_gen = Generator(in_channel=3, num_residual=9).to(config.device)

    disc_opt = torch.optim.Adam(list(mri_disc.parameters()) + list(ct_disc.parameters()), lr=config.lr,
                                betas=(0.5, 0.999))
    gen_opt = torch.optim.Adam(list(ct_gen.parameters()) + list(mri_gen.parameters()), lr=config.lr, betas=(0.5, 0.999))

    # train_dataset = CustomDataset(Config.TrainA_dir, Config.TrainB_dir, transform=Config.transforms)
    # train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    # valid_dataset = CustomDataset(Config.ValidA_dir, Config.ValidB_dir, transform=Config.transforms)
    # valid_loader = DataLoader(valid_dataset, batch_size=Config.batch_size, shuffle=False)

    L1 = torch.nn.L1Loss()
    mse = torch.nn.MSELoss()
    if config.load_model:
        load_checkpoint(config.checkpoint_gen_mri, mri_gen, gen_opt, config.lr)
        load_checkpoint(config.checkpoint_gen_ct, ct_gen, gen_opt, config.lr)
        load_checkpoint(config.checkpoint_gen_mri, mri_disc, disc_opt, config.lr)
        load_checkpoint(config.checkpoint_gen_ct, ct_disc, disc_opt, config.lr)

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    dll = []
    gll = []

    print('Start Training...')
    for epoch in range(config.epochs):
        print('Epoch [{}]'.format(epoch + 1))
        dl, gl = train(mri_disc, ct_disc, mri_gen, ct_gen, disc_opt, gen_opt, train_loader, valid_loader, mse, d_scaler,
                       g_scaler, L1, folder='Data/saved_img/eval')
        dll.append(dl)
        gll.append(gl)

        if config.save_model:
            save_checkpoint(mri_gen, gen_opt, filename=config.checkpoint_gen_mri)
            save_checkpoint(ct_gen, gen_opt, filename=config.checkpoint_gen_ct)
            save_checkpoint(mri_disc, disc_opt, filename=config.checkpoint_critic_mir)
            save_checkpoint(ct_disc, disc_opt, filename=config.checkpoint_critic_ct)
    plt.plot(dll, label='Discriminator Loss')
    plt.plot(gll, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
