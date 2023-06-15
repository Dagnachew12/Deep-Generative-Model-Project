import torch.cuda
from torchvision import transforms as T

device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 2e-4
batch_size = 16
epochs = 400
L1_LAMBDA = 100
lambda_gp = 1
BETA1 = 0.5

train_dir_MR = 'Data_1/MR'
train_dir_CT = 'Data_1/CT'

test_dir_MR = 'Data_sep/Test/MR'
test_dir_CT = 'Data_sep/Test/CT'


num_workers = 2
image_size = 256
load_model = False
save_model = True
checkpoint_gen_mri = 'gen_mri.pth.tar'
checkpoint_gen_ct = 'gen_ct.pth.tar'
checkpoint_critic_mir = 'critic_mir.pth.tar'
checkpoint_critic_ct = 'critic_ct.pth.tar'

transform = T.Compose(
    [
        T.ToPILImage(),
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
)

transforms = T.Compose(
    [
        T.ToPILImage(),
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ]
)


