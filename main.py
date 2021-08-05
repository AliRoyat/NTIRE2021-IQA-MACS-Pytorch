# PyTorch
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision.transforms as tr
import torch.nn.functional as F
from torch.nn import Sequential
# Models
from unet import Unet
from siamunet_conc import SiamUnet_conc
from SiamUnet_conc import SiamUnet_conc
from siamunet_diff import SiamUnet_diff

from fresunet import FresUNet

# Other
import os
import numpy as np
import random
from skimage import io
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
from IPython import display
from eval import eval
from accloss import accloss
from spearman import Spear

import time
import warnings

print('IMPORTS OK')

# Global Variables' Definitions

PATH_TO_DATASET = r'D:\NTIRE Workshop and Challenges @ CVPR 2021\dataset'

conf_type = "img(mean_std_norm)_label(mean_std_norm)_maxpool_1e-5_type_covcat_pretrain_epoch-134_loss-0.432_mse1_pr0.5_spr0.5"

sorter_checkpoint_path = r"D:\NTIRE Workshop and Challenges @ CVPR 2021\codes\FC-Siam-diff\fully_convolutional_change_detection-master\best_model0.00463445740044117.pth.tar"

BATCH_SIZE = 30

NUM_WORKER = 4

scale_co_test = 1

epoch_start_ = 21

N_EPOCHS = 200

NORMALISE_IMGS = True

NORMALISE_LABELS = True

TYPE = 2  # 0-RGB | 1-RGBIr | 2-All bands s.t. resulution <= 20m | 3-All bands

apply_spearman = True

LOAD_TRAINED = True
conf_type_pretrain = r"2080server/11"
if LOAD_TRAINED:
    ch_path = rf'./checkpoint/{conf_type}/ch_net-best_epoch-134_loss-0.4324578046798706.pth.tar'
    # ch_path = rf'./checkpoint/{conf_type}/ch_net-best_epoch-52_loss-1.37092924118042.pth.tar'
    # ch_path = rf'./checkpoint/{conf_type_pretrain}/ch_net-best_epoch-70_loss-0.6888124346733093.pth.tar'

DATA_AUG = True

print('DEFINITIONS OK')


def reshape_for_torch(I):
    """Transpose image for PyTorch coordinates."""
    #     out = np.swapaxes(I,1,2)
    #     out = np.swapaxes(out,0,1)
    #     out = out[np.newaxis,:]
    out = I.transpose((2, 0, 1))
    return torch.from_numpy(1.0 * out)


class NTIR(Dataset):
    """Change Detection dataset class, used for both training and test data."""

    def __init__(self, path, train=True, transform=None):

        self.transform = transform
        self.path = path
        self.names = [[], [], []]
        self.train_m = train

        if self.train_m:
            img_12 = 'path_img_train.txt'
            label = 'label_data_train.txt'

            with open(os.path.join(self.path, img_12), "r") as img_file:
                all_data = img_file.read().split("\n")[:-1]
                self.names[0] = [img.split(",")[0] for img in all_data]
                self.names[1] = [img.split(",")[1] for img in all_data]

            with open(os.path.join(self.path, label), "r") as gt_file:
                all_scores = gt_file.read().split("\n")[:-1]
                self.names[2] = [float(score) for score in all_scores]

            # self.names = [it[:200] for it in self.names]


        else:
            img_12 = 'path_img_test.txt'
            label = 'label_data_test.txt'

            with open(os.path.join(self.path, img_12), "r") as img_file:
                all_data = img_file.read().split("\n")[:-1]
                self.names[0] = [img.split(",")[0] for img in all_data]
                self.names[1] = [img.split(",")[1] for img in all_data]

            with open(os.path.join(self.path, label), "r") as gt_file:
                all_scores = gt_file.read().split("\n")[:-1]
                self.names[2] = [float(score) for score in all_scores]

            # self.names = [it[:200] for it in self.names]

    def __len__(self):
        return len(self.names[0])

    def __getitem__(self, idx):

        I1_path = self.names[0][idx]
        I2_path = self.names[1][idx]

        I1_ = io.imread(I1_path)
        I2_ = io.imread(I2_path)

        if NORMALISE_IMGS:
            I1_m = (I1_ - I1_.mean()) / I1_.std()
            I2_m = (I2_ - I2_.mean()) / I2_.std()
        else:
            I1_m = I1_
            I2_m = I2_
        I1 = reshape_for_torch(I1_m)
        I2 = reshape_for_torch(I2_m)

        label = np.array([self.names[2][idx]])
        if NORMALISE_LABELS:
            # label_ = label - np.array(self.names[2]).mean()
            label_ = (label - np.array(self.names[2]).mean()) / np.array(self.names[2]).std()
        else:
            label_ = label

        label = torch.from_numpy(1.0 * label_).float()

        sample = {'I1': I1, 'I2': I2, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


class RandomFlip(object):
    """Flip randomly the images in a sample."""

    #     def __init__(self):
    #         return

    def __call__(self, sample):
        I1, I2, label = sample['I1'], sample['I2'], sample['label']

        if random.random() > 0.5:
            I1 = I1.numpy()[:, :, ::-1].copy()
            I1 = torch.from_numpy(I1)
            I2 = I2.numpy()[:, :, ::-1].copy()
            I2 = torch.from_numpy(I2)

        return {'I1': I1, 'I2': I2, 'label': label}


class RandomRot(object):
    """Rotate randomly the images in a sample."""

    #     def __init__(self):
    #         return

    def __call__(self, sample):
        I1, I2, label = sample['I1'], sample['I2'], sample['label']

        n = random.randint(0, 3)
        if n:
            I1 = sample['I1'].numpy()
            I1 = np.rot90(I1, n, axes=(1, 2)).copy()
            I1 = torch.from_numpy(I1)
            I2 = sample['I2'].numpy()
            I2 = np.rot90(I2, n, axes=(1, 2)).copy()
            I2 = torch.from_numpy(I2)

        return {'I1': I1, 'I2': I2, 'label': label}


print('UTILS OK')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# net.load_state_dict(torch.load('net-best_epoch-1_fm-0.7394933126157746.pth.tar'))


def train(epock_start, best_lss, best_acc, n_epochs=N_EPOCHS, save=True):
    t = np.linspace(1, n_epochs, n_epochs)

    epoch_train_loss = 0 * t
    epoch_train_accuracy = 0 * t

    epoch_test_loss = 0 * t
    epoch_test_accuracy = 0 * t

    plt.figure(num=1)
    plt.figure(num=2)

    try:
        epoch_train_loss[:epock_start] = Train_loss_curve[1]
        epoch_test_loss[:epock_start] = Test_loss_curve[1]
        epoch_train_accuracy[:epock_start] = Train_accuracy_curve[1]
        epoch_test_accuracy[:epock_start] = Test_accuracy_curve[1]
    except:
        epoch_train_loss[:epock_start] = 0 * np.array(list(range(epock_start)))
        epoch_test_loss[:epock_start] = 0 * np.array(list(range(epock_start)))
        epoch_train_accuracy[:epock_start] = 0 * np.array(list(range(epock_start)))
        epoch_test_accuracy[:epock_start] = 0 * np.array(list(range(epock_start)))

    for epoch_index in tqdm(range(epock_start, n_epochs)):
        net.train()
        print('Epoch: ' + str(epoch_index + 1) + ' of ' + str(N_EPOCHS))

        for index, batch in enumerate(train_loader):
            # im1 = batch['I1'][2, :, :, :].numpy().transpose(1, 2, 0).astype("uint8")
            # im2 = batch['I2'][2, :, :, :].numpy().transpose(1, 2, 0).astype("uint8")
            I1 = Variable(batch['I1'].float().cuda())
            I2 = Variable(batch['I2'].float().cuda())
            label = Variable(batch['label'].cuda())

            optimizer.zero_grad()
            output = net(I1, I2)

            # loss = coef_loss_mae * criterion_mae(output, label) + coef_loss_pr * (
            #         1 - (criterion_pr(output, label)) ** 2)

            # zipp_sort_ind = zip(np.argsort(batch['label'].numpy())[::-1], range(BATCH_SIZE))
            # ranks = [((y[1] + 1) / float(BATCH_SIZE)) for y in sorted(zipp_sort_ind, key=lambda x: x[0])]
            # label_spr = torch.FloatTensor(ranks).cuda()

            loss = coef_loss_mse * criterion_mse(output, label) + coef_loss_pr * (1-criterion_pr(output,
                                                                                              label)) + coef_loss_spr * criterion_spr(
                output, label) + coef_loss_mae * criterion_mae(output, label)
            with torch.no_grad():
                print("@@@@",criterion_mse(output, label)," ", 1-criterion_pr(output,label), " ", criterion_spr(output, label))

            # loss = criterion_mse(output, label)

            loss.backward()
            optimizer.step()
            print(
                "\ntrain : " + f"epoch : {epoch_index + 1} -- " + f"{index + 1}" + " / " + f"{len(train_loader)}" + " ----->" + "loss : "
                + f"{loss.detach().cpu().numpy():.04f}")

        scheduler.step()
        with torch.no_grad():

            epoch_train_loss[epoch_index], epoch_train_accuracy[epoch_index] = test(train_loader_val)

            epoch_test_loss[epoch_index], epoch_test_accuracy[epoch_index] = test(test_loader)

        plt.figure(num=1)
        plt.clf()

        l1_1, = plt.plot(t[:epoch_index + 1], epoch_train_loss[:epoch_index + 1],
                         label='Train loss')
        l1_2, = plt.plot(t[:epoch_index + 1], epoch_test_loss[:epoch_index + 1],
                         label='Test loss')
        plt.legend(handles=[l1_1, l1_2])
        plt.grid()

        plt.gcf().gca().set_xlim(left=0)
        plt.title('Loss')
        display.clear_output(wait=True)
        display.display(plt.gcf())

        plt.figure(num=2)
        plt.clf()

        l2_1, = plt.plot(t[:epoch_index + 1], epoch_train_accuracy[:epoch_index + 1],
                         label='Train accuracy')
        l2_2, = plt.plot(t[:epoch_index + 1], epoch_test_accuracy[:epoch_index + 1],
                         label='Test accuracy')
        plt.legend(handles=[l2_1, l2_2])
        plt.grid()

        plt.gcf().gca().set_xlim(left=0)
        plt.title('Accuracy')
        display.clear_output(wait=True)
        display.display(plt.gcf())

        # lss = epoch_train_loss[epoch_index]
        # accu = epoch_train_accuracy[epoch_index]

        lss = epoch_test_loss[epoch_index]
        accu = epoch_test_accuracy[epoch_index]

        if accu > best_acc:
            best_acc = accu
            save_str = fr'./checkpoint/{conf_type}/ch_net-best_epoch-' + str(epoch_index + 1) + '_accu-' + str(
                accu) + '.pth.tar'
            # torch.save(net.state_dict(), save_str)

            torch.save({
                'epoch': epoch_index,
                'model_state_dict': net.state_dict(),
                'model_state_dict_head': criterion_spr.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                "Train loss": [t[:epoch_index + 1], epoch_train_loss[:epoch_index + 1]],
                "Test loss": [t[:epoch_index + 1], epoch_test_loss[:epoch_index + 1]],
                "Train accuracy": [t[:epoch_index + 1],
                                   epoch_train_accuracy[:epoch_index + 1]],
                "Test accuracy": [t[:epoch_index + 1], epoch_test_accuracy[:epoch_index + 1]],
                'loss': lss,
                'acc': accu
            }, save_str)

        if lss < best_lss:
            best_lss = lss
            save_str = rf'./checkpoint/{conf_type}/ch_net-best_epoch-' + str(epoch_index + 1) + '_loss-' + str(
                lss) + '.pth.tar'
            torch.save({
                'epoch': epoch_index,
                'model_state_dict': net.state_dict(),
                'model_state_dict_head': criterion_spr.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                "Train loss": [t[:epoch_index + 1], epoch_train_loss[:epoch_index + 1]],
                "Test loss": [t[:epoch_index + 1], epoch_test_loss[:epoch_index + 1]],
                "Train accuracy": [t[:epoch_index + 1],
                                   epoch_train_accuracy[:epoch_index + 1]],
                "Test accuracy": [t[:epoch_index + 1], epoch_test_accuracy[:epoch_index + 1]],
                'loss': lss,
                'acc': accu
            }, save_str)
        print(
            f"\n ################## \n epock : {epoch_index + 1} \n avg_loss_train : {lss} \n avg_acc_train : {accu} \n ################## \n")

        accu_val = epoch_test_accuracy[epoch_index]
        lss_val = epoch_test_loss[epoch_index]
        print(
            f"\n ################## \n epock : {epoch_index + 1} \n avg_loss_test : {lss_val} \n avg_acc_test : {accu_val} \n ################## \n")

        if save:
            im_format = 'png'
            #         im_format = 'eps'

            plt.figure(num=1)
            plt.savefig(net_name + '-01-loss.' + im_format)

            plt.figure(num=2)
            plt.savefig(net_name + '-02-accuracy.' + im_format)

    out = {'train_loss': epoch_train_loss[-1],
           'train_accuracy': epoch_train_accuracy[-1],
           'test_loss': epoch_test_loss[-1],
           'test_accuracy': epoch_test_accuracy[-1]}

    return out


L = 1024
N = 2


def test(dset):
    net.eval()
    tot_loss = 0
    tot_count = 0
    all_predicted = []
    all_gt = []

    for index, batch in enumerate(dset):
        I1 = Variable(batch['I1'].float().cuda())
        I2 = Variable(batch['I2'].float().cuda())
        cm = Variable(batch['label'].cuda())

        output = net(I1, I2)
        # loss = coef_loss_mae * criterion_mae(output, label) + coef_loss_pr * (
        #         1 - (criterion_pr(output, label)) ** 2)

        # zipp_sort_ind = zip(np.argsort(batch['label'].numpy())[::-1], range(BATCH_SIZE))
        # ranks = [((y[1] + 1) / float(BATCH_SIZE)) for y in sorted(zipp_sort_ind, key=lambda x: x[0])]
        # label_spr_cm = torch.FloatTensor(ranks).cuda()

        loss = coef_loss_mse * criterion_mse(output, cm) + coef_loss_pr * (1-criterion_pr(output,
                                                                                       cm)) + coef_loss_spr * criterion_spr(
            output, cm) + coef_loss_mae * criterion_mae(output, cm)
        # loss = criterion_mse(output, cm)

        print(
            "\n val : " + f"{index + 1}" + " / " + f"{len(dset)}" + " ----->" + "loss : "
            + f"{loss.detach().cpu().numpy():.04f}")
        tot_loss += loss.data * np.prod(cm.size())
        tot_count += np.prod(cm.size())
        all_predicted.extend(list(torch.squeeze(output).detach().cpu().numpy()))
        all_gt.extend(list(torch.squeeze(cm).detach().cpu().numpy()))

    net_loss = tot_loss / tot_count
    accuracy, _, _ = eval(np.array(all_predicted), np.array(all_gt))

    return net_loss, accuracy


def save_test_results(dset):
    for name in tqdm(dset.names):
        with warnings.catch_warnings():
            I1, I2, cm = dset.get_img(name)
            I1 = Variable(torch.unsqueeze(I1, 0).float()).cuda()
            I2 = Variable(torch.unsqueeze(I2, 0).float()).cuda()
            out = net(I1, I2)
            _, predicted = torch.max(out.data, 1)
            I = np.stack((255 * cm, 255 * np.squeeze(predicted.cpu().numpy()), 255 * cm), 2)
            io.imsave(f'{net_name}-{name}.png', I)


if __name__ == '__main__':

    if DATA_AUG:
        data_transform = tr.Compose([RandomFlip(), RandomRot()])
    else:
        data_transform = None

    train_dataset = NTIR(PATH_TO_DATASET, train=True, transform=data_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER,
                              drop_last=True)

    train_loader_val = DataLoader(train_dataset, batch_size=int(BATCH_SIZE * scale_co_test), shuffle=True,
                                  num_workers=NUM_WORKER,
                                  drop_last=True)

    test_dataset = NTIR(PATH_TO_DATASET, train=False)
    test_loader = DataLoader(test_dataset, batch_size=int(BATCH_SIZE * scale_co_test), shuffle=True,
                             num_workers=NUM_WORKER,
                             drop_last=True)

    print('DATASETS OK')

    if TYPE == 0:
        #     net, net_name = Unet(2*3, 2), 'FC-EF'
        #     net, net_name = SiamUnet_conc(3, 2), 'FC-Siam-conc'
        #     net, net_name = SiamUnet_diff(3, 2), 'FC-Siam-diff'
        net, net_name = FresUNet(2 * 3, 2), 'FresUNet'
    elif TYPE == 1:
        #     net, net_name = Unet(2*4, 2), 'FC-EF'
        #     net, net_name = SiamUnet_conc(4, 2), 'FC-Siam-conc'
        #     net, net_name = SiamUnet_diff(4, 2), 'FC-Siam-diff'
        net, net_name = FresUNet(2 * 4, 2), 'FresUNet'
    elif TYPE == 2:
        #     net, net_name = Unet(2*10, 2), 'FC-EF'
        net, net_name = SiamUnet_conc(3, 1), rf'./checkpoint/{conf_type}/FC-Siam-diff'
        #     net, net_name = SiamUnet_diff(10, 2), 'FC-Siam-diff'
        # net, net_name = FresUNet(2 * 10, 2), 'FresUNet'
    elif TYPE == 3:
        #     net, net_name = Unet(2*13, 2), 'FC-EF'
        #     net, net_name = SiamUnet_conc(13, 2), 'FC-Siam-conc'
        net, net_name = SiamUnet_diff(3, 1), rf'./checkpoint/{conf_type}/FC-Siam-diff'
        # net, net_name = FresUNet(2 * 13, 2), 'FresUNet'

    net.cuda()

    criterion_mse = F.mse_loss
    criterion_mae = F.l1_loss
    criterion_pr = accloss


    coef_loss_mse = 1
    coef_loss_mae = 0
    coef_loss_pr = 0.5
    coef_loss_spr = 0.5



    print('Number of trainable parameters:', count_parameters(net))

    print('NETWORK OK')

    #     optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)

    if LOAD_TRAINED:

        checkpoint = torch.load(ch_path)
        # checkpoint['optimizer_state_dict']['param_groups'][0]["lr"] = 1e-5
        try:
            net.load_state_dict(checkpoint['model_state_dict'])
            criterion_spr = Spear(sorter_checkpoint_path)
            optimizer = torch.optim.Adam(list(net.parameters()) + list(criterion_spr.parameters()), lr=1e-5,
                                         weight_decay=1e-4)
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
            # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            # Train_loss_curve = checkpoint["Train loss"]
            # Test_loss_curve = checkpoint["Test loss"]
            # Train_accuracy_curve = checkpoint["Train accuracy"]
            # Test_accuracy_curve = checkpoint["Test accuracy"]
            # epoch_input = checkpoint['epoch'] + 1
            # best_acc_ = checkpoint['epoch']
            # best_lss_ = checkpoint['epoch']
            epoch_input = 0
            best_acc_ = 0
            best_lss_ = 1000

        except:
            epoch_input = epoch_start_
            net.load_state_dict(checkpoint)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
            best_acc_ = 0
            best_lss_ = 1000
    else:
        epoch_input = 0
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
        best_acc_ = 0
        best_lss_ = 1000

    print('LOAD OK')

    t_start = time.time()
    out_dic = train(epoch_input, best_lss_, best_acc_)
    t_end = time.time()
    print(out_dic)
    print('Elapsed time:')
    print(t_end - t_start)

    if not LOAD_TRAINED:
        torch.save(net.state_dict(), rf'./checkpoint/{conf_type}/net_final.pth.tar')
        print('SAVE OK')
    import pdb
    pdb.tra
# t_start = time.time()
# # save_test_results(train_dataset)
# save_test_results(test_dataset)
# t_end = time.time()
# print('Elapsed time: {}'.format(t_end - t_start))
