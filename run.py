import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

DEVICE = 'cuda' if torch.cuda.is_available else 'cpu'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def viz(img, flo, i):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)
    
    plt.imshow(img_flo / 255.0)
    plt.savefig(f'/home/sharif/Documents/RAFT/test_vis/{i}.png')

    # clear plt
    plt.clf()
    plt.cla()

def run(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load('models/raft-things.pth'))

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_dir = Path(args.output_dir)
    images_dir = Path(args.images_dir)
    images = list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg'))

    with torch.no_grad():
        images = sorted(images)

        for i in range(len(images)-1):
            im_f1 = str(images[i])
            im_f2 = str(images[i+1])
            
            image1 = load_image(im_f1)
            image2 = load_image(im_f2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
                
            # 2.2 MB
            of_f_name = output_dir / f'{i}.npy' 
            np.save(of_f_name, flow_up.cpu())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', help="directory with your images")
    parser.add_argument('--output_dir', help="optical flow images will be stored here as .npy files")
    args = parser.parse_args()

    run(args)
