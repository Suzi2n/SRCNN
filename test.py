import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
import os

from models import SRCNN
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr
from datasets import TestDataset
from torch.utils.data.dataloader import DataLoader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #가중치 파일 경로 (최고 성능을 갖는 모델을 불러옴)
    parser.add_argument('--weights-file', type=str, default='./outputs/best.pth')
    # 가중치 파일 경로 (3번째 에포크 모델을 불러옴)
    #parser.add_argument('--weights-file', type=str, default='./outputs/epoch_0.pth')
    # 테스트 데이터셋 경로
    parser.add_argument('--test-dir', type=str, default='./db/test')
    # 출력 영상 경로
    parser.add_argument('--outimg-dir', type=str, default='outimg/')
    # 입력 영상 경로
    parser.add_argument('--orgimg-dir', type=str, default='orgimg/')
    parser.add_argument('--scale', type=int, default=4)
    args = parser.parse_args()

    if os.path.exists(args.orgimg_dir) == False:
        os.mkdir(args.orgimg_dir)
    if os.path.exists(args.outimg_dir) == False:
        os.mkdir(args.outimg_dir)
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = SRCNN().to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    test_dataset = TestDataset(args.test_dir, scale=args.scale)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    model.eval()

    img_num = len(test_dataset.hr_list)
    psnr = torch.zeros(img_num)

    n = -1
    for data in test_dataloader:
        n = n + 1
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            preds = model(inputs).clamp(0.0, 1.0)
        psnr[n] = calc_psnr(preds, labels)
        print('{:d}/{:d}    PSNR: {:.2f}'.format(n+1, len(test_dataset), psnr[n]))

        preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

        image = pil_image.open(test_dataset.hr_list[n]).convert('RGB')
        image_width = (image.width // args.scale) * args.scale
        image_height = (image.height // args.scale) * args.scale
        image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
        image = image.resize((image.width // args.scale, image.height // args.scale), resample=pil_image.BICUBIC)
        image = image.resize((image.width * args.scale, image.height * args.scale), resample=pil_image.BICUBIC)


        _, fname = os.path.split(test_dataset.hr_list[n])
        filename = "{}/{}".format(args.orgimg_dir, fname)
        image.save(filename)

        image = np.array(image).astype(np.float32)
        ycbcr = convert_rgb_to_ycbcr(image)

        output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
        output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
        output = pil_image.fromarray(output)

        # split
        _, fname = os.path.split(test_dataset.hr_list[n])
        filename = "{}/{}".format(args.outimg_dir, fname)
        output.save(filename)
    mean_psnr = torch.mean(psnr)
    print('Total PSNR: {:.2f}'.format(mean_psnr))
