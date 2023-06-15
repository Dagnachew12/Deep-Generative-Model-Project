import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import config
from pix2pixGAN import Discriminator, Generator
from torch import nn
from Dataset import CustomDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from Eval import *
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from util import *
from Eval import *


# from Validation import *


def train(gen_net, disc_net, train_dl, valid_dl, gen_opt, disc_opt, l1_loss, bce_loss, folder):
    pnsr = PeakSignalNoiseRatio().to(config.device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(config.device)
    loss = nn.MSELoss()
    loop = tqdm(train_dl, leave=True)
    psnr_avg = []
    ssmi_avg = []
    mse_avg = []
    for i, (mr, ct) in enumerate(loop):
        mr = mr.to(config.device)
        ct = ct.to(config.device)

        ##################################
        #          TRAIN GENERATOR       #
        ##################################
        ct_fake = gen_net(mr)
        D_fake = disc_net(ct_fake)
        D_real = disc_net(ct)
        G_fake_loss = bce_loss(D_fake, torch.ones_like(D_fake))
        # G_real_loss = bce_loss(D_real, torch.ones_like(D_real))
        # G_loss = (G_real_loss + G_fake_loss) / 2
        L1 = l1_loss(ct_fake, ct)
        G_loss = L1 * config.L1_LAMBDA + G_fake_loss
        gen_opt.zero_grad()
        G_loss.backward()
        gen_opt.step()

        ##################################
        #      TRAIN DISCRIMINATOR       #
        ##################################
        ct_fake = gen_net(mr)
        D_real = disc_net(ct)
        D_real_loss = bce_loss(D_real, torch.ones_like(D_real))
        D_fake = disc_net(ct_fake)
        D_fake_loss = bce_loss(D_fake, torch.zeros_like(D_fake))
        D_loss = (D_real_loss + D_fake_loss) / 2
        disc_opt.zero_grad()
        D_loss.backward()
        disc_opt.step()

        if i % 2 == 0:
            save_image(mr * 0.5 + 0.5, f'Test_data/Train/mri_{i}.png')
            save_image(ct_fake * 0.5 + 0.5, f"Test_data/Train/fake_ct_{i}.png")
            save_image(ct * 0.5 + 0.5, f"Test_data/Train/ct_{i}.png")
            # save_image(fake_ct * 0.5 + 0.5, f"saved_images/fake_ct_{i}.png")

        if i % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item()
            )
        # ssim = SSIM(data_range=1.0).to(config.device)
        # ssim = ssim(ct, ct_fak)
    for j, (mrv, ctt) in enumerate(valid_dl):
        mrv = mrv.to(config.device)
        ctt = ctt.to(config.device)
        gen_net.eval()
        with torch.no_grad():
            Fake_ct = gen_net(mrv)
            mse = loss(ctt * 0.5 + 0.5, Fake_ct * 0.5 + 0.5)
            pnsr_val = pnsr(ctt * 0.5 + 0.5, Fake_ct * 0.5 + 0.5)
            ssimm = ssim(ctt * 0.5 + 0.5, Fake_ct * 0.5 + 0.5)
            Fake_ct = Fake_ct * 0.5 + 0.5
            img_er = torch.abs((ctt * 0.5 + 0.5) - Fake_ct)
            psnr_avg.append(pnsr_val.cpu())
            ssmi_avg.append(ssimm.cpu())
            mse_avg.append(mse.cpu())
            if j % 1 == 0:
                save_image(mrv * 0.5 + 0.5, folder + f"/mri{j}.png")
                save_image(ctt * 0.5 + 0.5, folder + f"/ct{j}.png")
                save_image(Fake_ct, folder + f"/fake_ct{j}.png")
                save_image(img_er, folder + f"/ct_er{j}.png")
        gen_net.train()
    print(
        'Disc Loss: {:.6f}  Gen Loss: {:.6f} PNSR:{:.6f} SSIM: {:.6f} MSE: {:.6f}'.format(D_loss.item(), G_loss.item(),
                                                                                          np.array(psnr_avg).mean(), np.array(ssmi_avg).mean(), np.array(mse_avg).mean()))
    # print('Disc Loss: {}  '.format(D_loss.item()))


dataset = CustomDataset(config.train_dir_MR, config.train_dir_CT, transform=config.transform)
train_dataset, val_dataset = train_test_split(dataset, test_size=0.1, random_state=42)
valid_dataset, test_dataset = train_test_split(val_dataset, test_size=0.5, random_state=42)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False)
# test_dataset = ImageDataset(config.test_dir_MR, config.test_dir_CT, transform=config.transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


def run():
    # disc_net = Discriminator(in_channel=3).to(config.device)
    # gen_net = Generator(in_channels=3).to(config.device)
    disc_net = Discriminator(3).to(config.device)
    gen_net = Generator().to(config.device)
    disc_opt = torch.optim.Adam(disc_net.parameters(), lr=config.lr, betas=(config.BETA1, 0.999))
    gen_optim = torch.optim.Adam(gen_net.parameters(), lr=config.lr, betas=(config.BETA1, 0.999))

    bce_loss = nn.BCEWithLogitsLoss()
    L1_loss = nn.L1Loss()

    if config.load_model:
        load_checkpoint(config.checkpoint_gen_mri, gen_net, gen_optim, config.lr)
        load_checkpoint(config.checkpoint_gen_ct, gen_net, gen_optim, config.lr)
        load_checkpoint(config.checkpoint_critic_ct, disc_net, disc_opt, config.lr)
        load_checkpoint(config.checkpoint_critic_mir, disc_net, disc_opt, config.lr)

    print('Start Training...')
    for epoch in range(config.epochs):
        print('Epoch [{}]'.format(epoch + 1))
        train(gen_net, disc_net, train_loader, valid_loader, gen_optim, disc_opt, L1_loss, bce_loss, folder='Test_data/Eval')
        # save_examples(ct_gen, valid_loader, folder='Evaluation')
        if config.save_model:
            save_checkpoint(gen_net, gen_optim, filename=config.checkpoint_gen_mri)
            save_checkpoint(gen_net, gen_optim, filename=config.checkpoint_gen_ct)
            save_checkpoint(disc_net, disc_opt, filename=config.checkpoint_critic_mir)
            save_checkpoint(disc_net, disc_opt, filename=config.checkpoint_critic_ct)


if __name__ == '__main__':
    # run()
    eval_image(ct_gen, test_loader, folder='Test_data/Test/com')