import argparse
import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import PIL
import archs
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter
import archtyr2


import albumentations as A
"""
需要指定参数：--name dsb2018_96_NestedUNet_woDS
"""

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default="AttentionTransformUnet++",
                        help='model name')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print(args.name)
    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    if (config['arch'] == 'UnetPlusPlus'):
        print('1')
        model = archtyr2.UNetPlusPlus(in_channels =3,out_channels =4)
    else:
        model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision']
                                          )

    model = model.cuda()

    # Data loading code
    #img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    print(config['dataset_train_image'][:43]+'img_test')
    img_ids = glob(os.path.join(config['dataset_train_image'][:43]+'img_test', '*' + config['img_ext']))
    val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    # img_ids = glob(os.path.join(config['dataset'].split('/')[:4]+'/mask_test', '*' + config['mask_ext']))
    # img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    # _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    model.load_state_dict(torch.load('models/%s/model.pth' %
                                     config['name']))
    model.eval()

    # val_transform = Compose([
    #     A.Resize(config['input_h'], config['input_w']),
    #     transforms.Normalize(),
    # ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=config['dataset_train_image'][:43]+'img_test',
        mask_dir=config['dataset_train_image'][:43]+'/mask_test',
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        # num_classes=config['num_classes'],
        # transform=val_transform)
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    # avg_meters = {'iou': AverageMeter()}
    avg_meters = AverageMeter()
    headIou_avg = AverageMeter()
    bodyIou_avg = AverageMeter()
    footIou_avg = AverageMeter()

    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    iou_epoch = 0
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                output = model(input)[-1]
            else:
                output = model(input)
            # print(output.shape)

            # output = output.cpu()
            # target = target.cpu()
            # print("Output shape:", output.shape)    期望为 [batch_size, height, width]
            # print("Target shape:", target.shape)    期望为 [batch_size, height, width]
            # print('11111111',input.shape)
            iou, footIou, bodyIou, headIou = iou_score(output, target)
            # print('1111111111111111','output:',output.shape,'target:',target.shape,'input:',input.shape)
            # iou = iou / input.size(0)
            
            # iou_epoch += iou
            avg_meters.update(iou, input.size(0))
            footIou_avg.update(footIou, input.size(0))
            bodyIou_avg.update(bodyIou, input.size(0))
            headIou_avg.update(headIou, input.size(0))
            # avg_meters['loss'].update(loss.item(), input.size(0))
            # avg_meters['iou'].update(iou, input.size(0))


            #  for i in range(len(output)):
            #     for c in range(config['num_classes']):
            #         cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),
            #                     (output[i, c] * 255).astype('uint8')
            #                     # (output[i, c] * 255*85).astype('uint8')
            #                    )
    
            # for i in range(len(output_classes)):  # 遍历每个图像
            #     for c in range(config['num_classes']):  # 遍历每个类别
            #     # 生成当前类别的二值化图像
            #         binary_mask = (output_classes[i] == c).astype('uint8') * 255  # 将类别 c 转换为二值图像
            #         # 保存图像
            #         cv2.imwrite(
            #             os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),
            #             binary_mask
            #         )

            class_to_gray = {0: 0, 1: 85, 2: 170, 3: 255}

# 保存合并后的灰度图像
            for i in range(len(output)):  # 遍历每张图像
                # gray_image = np.zeros_like(output[i], dtype=np.uint8)  # 创建一个空的灰度图像
                gray_image = np.zeros((output.shape[2], output.shape[3]), dtype=np.uint8)  # 转换到 CPU 后再创建灰度图像

                for cls, gray_val in class_to_gray.items():
                    # gray_image[output[i] == cls] = gray_val  # 将属于该类别的像素设为对应灰度值
                    gray_image[(output[i,cls].cpu().numpy() >= 0.6)] = gray_val
                    # print("Unique values in gray_image:", np.unique(gray_image))

                    # print(gray_image.shape)
                # if gray_image.ndim == 3 and gray_image.shape[2] == 1:
                #     print('qizuoyongle')
                #     gray_image = gray_image.squeeze(-1)  # 去掉最后一个维度

    # 保存灰度图像
                # print('11111',gray_image.shape)
                cv2.imwrite(
                    os.path.join('outputs', config['name'], meta['img_id'][i] + '.jpg'),
                    gray_image
                )
        
        # print(iou_epoch/len(val_loader))
    # print('IoU: %.4f, Iou_head:%.4f, Iou_body:%.4f, Iou_foot:%.4f' % (avg_meters.avg, iou_classes[1],iou_classes[2],iou_classes[3]))
    print(avg_meters.avg, footIou_avg.avg, bodyIou_avg.avg, headIou_avg.avg)
    
    
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
