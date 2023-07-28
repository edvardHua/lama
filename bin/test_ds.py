# -*- coding: utf-8 -*-
# @Time : 2023/7/25 12:00
# @Author : zihua.zeng
# @File : test_ds.py

import os, sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "saicinpainting"))

import hydra
from omegaconf import OmegaConf

from saicinpainting.training.trainers import make_training_model
from saicinpainting.training.data.datasets import make_default_train_dataloader, make_default_val_dataloader
from saicinpainting.img_util import tensor2img, imwrite


@hydra.main(config_path='../configs/training', config_name='tiny_test.yaml')
def main(config: OmegaConf):
    train_loader = make_default_train_dataloader(**config.data.train)
    # c = 0
    for i, item in enumerate(train_loader):
        image = item['image_logo']
        mask = item['mask']
        # out_image = tensor2img(image)
        # out_mask = tensor2img(mask)
        # imwrite(out_image, 'image_%d.png' % c)
        # imwrite(out_mask, 'mask_%d.png' % c)
        # c += 1
        # print(c)


if __name__ == "__main__":
    main()
