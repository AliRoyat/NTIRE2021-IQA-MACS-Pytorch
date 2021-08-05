import glob
import os
import numpy as np
from skimage import io
import random
from tqdm import tqdm
from torch.autograd import Variable
from siamunet_diff import SiamUnet_diff
from siamunet_conc import SiamUnet_conc
from checkpoint.cinavad_sever.siamunet_conc_extrahead import SiamUnet_conc
import torch
from img_ensamble import image_ensamble

# conf_type = r"img(mean_std_norm)_label(mean_std_norm)_maxpool_1e-5_type_covcat_pretrain_epoch-180_loss-0.42687_mse1_pr0.5_spr0.5"
conf_type = r"D:\NTIRE Workshop and Challenges @ CVPR 2021\results\60\img(mean_std_norm)_label(mean_std_norm)_maxpool_1e-5_type_covcat_pretrain_epoch-172_loss-0.575_mse1_pr1_spr1"
NORMALISE_IMGS = True
device = "cuda"
TYPE = "new"


def apply_img_to_net(img):
    return random.uniform(1260.125454, 1590.555546)


def reshape_for_torch(I):
    """Transpose image for PyTorch coordinates."""
    #     out = np.swapaxes(I,1,2)
    #     out = np.swapaxes(out,0,1)
    #     out = out[np.newaxis,:]
    out = I.transpose((2, 0, 1))
    out = np.expand_dims(out, axis=0)
    return torch.from_numpy(1.0 * out)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    if TYPE == 2:
        net = SiamUnet_conc(3, 1)
    elif TYPE == "new":
        net = SiamUnet_conc(3, 32)
    elif TYPE == 3:
        net = SiamUnet_diff(3, 1)
    if device == "cuda":
        net.cuda()
    # checkpoint_path = torch.load(rf'./checkpoint/{conf_type}/ch_net-best_epoch-37_accu-1.5132.pth.tar')
    checkpoint = torch.load(rf'{conf_type}\ch_net-best_epoch-389_loss-2.5346076488494873.pth.tar')
    net.load_state_dict(checkpoint['model_state_dict'])

    val_path = r"D:\NTIRE Workshop and Challenges @ CVPR 2021\dataset\Dis"
    save_path = r"D:\NTIRE Workshop and Challenges @ CVPR 2021\results"

    with torch.no_grad():
        net.eval()
        for img in tqdm(glob.glob(os.path.join(val_path, "*.bmp"))):
            ref = img.replace("Dis", "Ref")[:-10] + ".bmp"

            I1_ = io.imread(img)
            I2_ = io.imread(ref)

            if NORMALISE_IMGS:
                I1_m = (I1_ - I1_.mean()) / I1_.std()
                I2_m = (I2_ - I2_.mean()) / I2_.std()
            else:
                I1_m = I1_
                I2_m = I2_
            I1 = Variable(reshape_for_torch(I1_m).float().to(device))
            I2 = Variable(reshape_for_torch(I2_m).float().to(device))

            # predicted_score = net(I1, I2)
            avg_predicted_score = image_ensamble(I1, I2, net)
            std = 121.7751
            mean = 1448.9539
            # predicted_score = (torch.squeeze(predicted_score).detach().cpu().numpy())*std + mean
            predicted_score = (avg_predicted_score * std) + mean
            # predicted_score = avg_predicted_score + mean
            # predicted_score = avg_predicted_score
            with open(os.path.join(save_path, "output.txt"), "a") as out_file:
                out_file.write(img.split("\\")[-1] + "," + f"{predicted_score:.4f}" + "\n")
