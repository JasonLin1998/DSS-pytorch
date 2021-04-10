import torch
from dssnet import build_model, weights_init
from PIL import Image
from torch.nn import utils, functional as F
from torchvision import transforms
import os
import matplotlib.pyplot as plt
import pylab
import numpy as np
from skimage.transform import resize
from tools.crf_process import crf
import cv2

def load_model(pthfile):
    net_params = torch.load(pthfile)
    #print(dict(net_params))
    model = build_model()
    model.train()
    model.apply(weights_init)
    model.load_state_dict(net_params)
    return model
    #print(model)


def test_one_img(img_root, num, model, mask_root, use_crf=True):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    images = os.listdir(img_root)
    test_img = images[num]
    test_img_path = os.path.join(img_root, test_img)
    image = Image.open(test_img_path)
    image_resize = image.resize((256, 256))

    masks = os.listdir(mask_root)
    test_mask = masks[num]
    test_mask_path = os.path.join(mask_root, test_mask)
    mask = Image.open(test_mask_path)
    mask = transform(mask).unsqueeze(0)
    shape = mask.size()[2:]

    image_t = transform(image).unsqueeze(0)

    y_pred = model(image_t)

    y_show = torch.mean(torch.cat([y_pred[i] for i in [1, 2, 3, 6]], dim=1), dim=1, keepdim=True)
    prob_pred = F.interpolate(y_show, size=shape, mode='bilinear', align_corners=True).cpu().data
    #print(np.shape(prob_pred))
    y_show1 = prob_pred.cpu().data[0][0]
    #print(np.shape(y_show1))
    if use_crf:
        y = crf(image_resize, y_show1.numpy(), to_tensor=True)
        y = y.squeeze()
        y = y.numpy()
        y_show2 = y
    else:
        y_show2 = y_show1.squeeze()
        #print(np.shape(y_show2))
        y_show2 = y_show2.numpy()
        #print(np.shape(y_show3))

    plt.subplot(1, 2, 1)
    plt.imshow(y_show2, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(image_resize)
    pylab.show()
    return y_show2



if __name__ == '__main__':
    pthfile = './results/run-mist/models/epoch_600.pth'
    #pthfile1 = './weights/final.pth'
    img_root = './data/RGBT/image/RGB'
    #img_root1 = './data/MSRA-B/image'
    mask_root = './data/RGBT/annotation'
    model = load_model(pthfile)
    #print(model)
    output = test_one_img(img_root, 299, model, mask_root, use_crf=True)
    cv2.imwrite('./data/RGBT/image/output/034_out_DSS.png', output * 255)

