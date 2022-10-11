import torch

from basicsr.utils.registry import ARCH_REGISTRY
import os
import numpy as np
import torch.nn as nn

@ARCH_REGISTRY.register()
class RED_CNN(nn.Module):
    def __init__(self, out_ch=96):
        super(RED_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv5 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)

        self.tconv1 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv2 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv3 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv4 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv5 = nn.ConvTranspose2d(out_ch, 1, kernel_size=5, stride=1, padding=0)

        # self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        # encoder
        residual_1 = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        residual_2 = out
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        residual_3 = out
        out = self.relu(self.conv5(out))
        # decoder
        out = self.tconv1(out)
        out += residual_3
        out = self.tconv2(self.relu(out))
        out = self.tconv3(self.relu(out))
        out += residual_2
        out = self.tconv4(self.relu(out))
        out = self.tconv5(self.relu(out))
        out += residual_1
        out = self.relu(out)
        return out

def test():
    import pytorch_radon as pr
    from pytorch_wavelets import DWTForward, DWTInverse
    import numpy as np
    import cv2

    redcnn = RED_CNN()
    load_path = "/home/szy/Downloads/net_g_280000.pth"
    load = torch.load(load_path)["params"]
    redcnn.load_state_dict(load, True)


    ll_path = "/media/szy/20CAA339CAA309DC/data/red_cnn/ld/val/ll_1.npy"
    lh_path = "/media/szy/20CAA339CAA309DC/data/red_cnn/ld/val/lh_1.npy"
    hl_path = "/media/szy/20CAA339CAA309DC/data/red_cnn/ld/val/hl_1.npy"
    hh_path = "/media/szy/20CAA339CAA309DC/data/red_cnn/ld/val/hh_1.npy"
    ll_ld = np.load(ll_path, allow_pickle=True)*100
    lh_ld = np.load(lh_path, allow_pickle=True)*100
    hl_ld = np.load(hl_path, allow_pickle=True)*100
    hh_ld = np.load(hh_path, allow_pickle=True)*100

    out = []

    for name in [ll_ld, lh_ld, hl_ld, hh_ld]:
        tensor = torch.from_numpy(name).view(1, 1, 256, 540)
        out.append(redcnn(tensor) / 100)

    l = out[0]
    tmp = []
    for i in out[1:]:
        tmp.append(torch.unsqueeze(i, 2))
    h = torch.cat(tmp, 2)

    radon = pr.radon.Radon(in_size=512, theta=torch.arange(0, 1080)).cuda()
    iradon = pr.radon.IRadon(theta=torch.arange(0, 1080)).cuda()
    idwt = DWTInverse(wave="haar", mode="zero").cuda()

    img_ori_path = "/media/szy/20CAA339CAA309DC/data/HDCT/1.jpg"
    img_ori = cv2.imread(img_ori_path)
    img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY) / 255.
    img_ori_cuda = torch.from_numpy(img_ori).float().view(1, 1, 512, 512).cuda()
    img_gt = iradon(radon(img_ori_cuda)).detach().cpu().numpy()[0, 0, :, :]
    img_gt[img_gt > 1] = 1
    img_gt[img_gt < 0] = 0
    img_gt = np.uint8(img_gt*255)

    img_noise_path = "/media/szy/20CAA339CAA309DC/data/datanoise/9/1.jpg"
    img_noise = cv2.imread(img_noise_path)

    t = idwt((l.cuda(), [h.cuda()]))
    r = torch.log(1 / t)
    u = iradon(r)
    im = u * 1000
    img = im.detach().cpu().numpy()[0, 0, :, :]
    img_denoised = np.uint8((img - img.min()) / (img.max() - img.min()) * 255)

    cv2.imshow("gt", img_gt)
    cv2.imshow("noise", img_noise)
    cv2.imshow("denoise", img_denoised)
    cv2.waitKey(0)

if __name__ == '__main__':
    test()