import torch
from torch.utils.data import DataLoader
import config
from Dataset import CustomDataset
from torchvision.utils import save_image
# from Model.Pixl2PixlGAN import Generator, Discriminator
from pix2pixGAN import Generator, Discriminator
from util import *
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from Train import *

# valid_dataset = ImageDataset(config.valid_dir_MR, config.valid_dir_CT, transform=config.transforms)
# valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False)
#
# mri_gen = Generator(in_channels=3).to(config.device)
# ct_gen = Generator(in_channels=3).to(config.device)
mri_gen = Generator().to(config.device)
ct_gen = Generator().to(config.device)
ct_opt = torch.optim.Adam(ct_gen.parameters(), lr=config.lr, betas=(0.5, 0.999))

# utils.load_checkpoint(config.checkpoint_gen_mri, mri_gen, gen_opt, config.lr)
# utils.load_checkpoint(config.checkpoint_gen_ct, ct_gen, ct_opt, config.lr)


def eval_image(ctgen, loader, folder):
    load_checkpoint(config.checkpoint_gen_ct, ct_gen, ct_opt, config.lr)
    for i, (mri, ct) in enumerate(loader):
        pnsr = PeakSignalNoiseRatio().to(config.device)
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(config.device)
        loss = torch.nn.MSELoss()
        ct = ct.to(config.device)
        mri = mri.to(config.device)
        ctgen.eval()
        with torch.no_grad():
            Fake_ct = ctgen(mri)
            Fake_ct = Fake_ct * 0.5 + 0.5
            pnsrt_val = pnsr(ct * 0.5 + 0.5, Fake_ct)
            ssimt = ssim(ct * 0.5 + 0.5, Fake_ct)
            mse = loss(ct * 0.5 + 0.5, Fake_ct).item()
            ct_error = torch.abs((ct * 0.5 + 0.5) - Fake_ct)
            if i % 1 == 0:
                save_image(mri * 0.5 + 0.5, folder + f"/mri{i}.png")
                save_image(ct * 0.5 + 0.5, folder + f"/ct{i}.png")
                save_image(Fake_ct, folder + f"/fake_ct{i}.png")
                save_image(ct_error, folder + f"/ct_er{i}.png")
                print('PSNR: {}  SSIM:{}  MSE: {}'.format(pnsrt_val, ssimt, mse))
        ctgen.train()

