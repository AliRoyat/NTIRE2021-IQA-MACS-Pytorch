import numpy as np
import torch

device = "cuda"


def reshape_for_torch(I):
    """Transpose image for PyTorch coordinates."""
    #     out = np.swapaxes(I,1,2)
    #     out = np.swapaxes(out,0,1)
    #     out = out[np.newaxis,:]
    out = I.transpose((2, 0, 1))
    out = np.expand_dims(out, axis=0)
    return torch.from_numpy(1.0 * out)


def image_ensamble(I1_, I2_, net):
    all_predicted_score = []

    for n in [0, 1, 2, 3]:
        for direction in ["none", "v", "h"]:
            I1, I2 = RandomFlip(I1_, I2_, direction)
            I1, I2 = RandomRot(I1, I2, n)
            predicted_score = net(I1, I2)
            predicted_score_ = torch.squeeze(predicted_score).detach().cpu().numpy()
            all_predicted_score.append(predicted_score_)
    avg_score = np.array(all_predicted_score).mean()
    return avg_score


def RandomRot(I1_, I2_, n):
    I1 = I1_[0, ...]
    I2 = I2_[0, ...]

    I1 = I1.cpu().numpy().copy()
    I1 = np.rot90(I1, n, axes=(1, 2)).copy()
    I1 = np.expand_dims(I1, axis=0)
    I1 = torch.from_numpy(I1).to(device)

    I2 = I2.cpu().numpy().copy()
    I2 = np.rot90(I2, n, axes=(1, 2)).copy()
    I2 = np.expand_dims(I2, axis=0)
    I2 = torch.from_numpy(I2).to(device)
    
    return I1, I2


def RandomFlip(I1_, I2_, direction):
    I1 = I1_
    I2 = I2_
    if direction == "none":
        return I1_, I2_
    elif direction == "v":
        I1 = I1.cpu().numpy()[:, :, :, ::-1].copy()
        I1 = torch.from_numpy(I1).to(device)
        I2 = I2.cpu().numpy()[:, :, :, ::-1].copy()
        I2 = torch.from_numpy(I2).to(device)

    elif direction == "h":
        I1 = I1.cpu().numpy()[:, :, ::-1, :].copy()
        I1 = torch.from_numpy(I1).to(device)
        I2 = I2.cpu().numpy()[:, :, ::-1, :].copy()
        I2 = torch.from_numpy(I2).to(device)

    return I1, I2
