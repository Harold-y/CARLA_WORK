import functools
import json
import os
import random
import shutil
from collections import defaultdict

import matplotlib.pyplot as plt
import pycocotools
from pycocotools import mask
import numpy as np
import pandas as pd
import uuid
import cv2
from pprint import pprint

import itertools

from PIL import Image

LABEL_INTER_PATH = 'D:\Programming\PycharmProjects\\resources\Kitti\Kitti\\raw\\training\\label_2'
# img translation: origin_size -> crop top 120 px ->  rescale to TARGET_IMG_SIZE
TARGET_IMG_SIZE = (1024, 256)

CROP_SIZE = 120

categories = set()
categories_dist = defaultdict(int)
fixed_category = {
    k: idx for idx, k in
    list(zip([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
             ['Cyclist', 'Van', 'Car', 'Misc', 'Pedestrian', 'Tram', 'Truck', 'DontCare', 'Person', 'Person_sitting']))
    # list(zip([0, 1, 2], ['Car', 'Van', 'Truck', ]))
}


def rescale_real_img(img_list):
    """
    crop and rescale KITTI image to TARGET size
    :param img_list:
    :return:
    """
    for idx, img_name in enumerate(sorted(img_list)):
        img_path = os.path.join(img_root, img_name)
        img = cv2.imread(img_path)[CROP_SIZE:, ...]
        img = cv2.resize(img, TARGET_IMG_SIZE)
        cv2.imwrite(
            os.path.join('../../resources/results/real/images/', img_name.replace('/', '_')),
            img
        )
        # cv2.imshow('', img)
        # cv2.waitKey(0)
        print(img_path)


def KITTI_tracking_to_intermediate(vis=False, ):
    if os.path.exists(LABEL_INTER_PATH): shutil.rmtree(LABEL_INTER_PATH)
    os.mkdir(LABEL_INTER_PATH)
    root = 'D:\Programming\PycharmProjects\\resources\Kitti\Kitti\\raw\\training\{}'
    # bounding box
    for path in map(functools.partial(os.path.join, root.format('label_2')),
                    sorted(os.listdir(root.format('label_2')))):
        labels = pd.read_csv(path, delimiter=' ', header=None).values[:, (0, 2, 6, 7, 8, 9)]
        labels[:, (3, 5)] -= np.minimum(labels[:, (3, 5)],
                                        CROP_SIZE)  # 偏移bb 裁剪图像上方CROP_SIZEpx的内容 小于CROP_SIZEpx的直接裁剪原bbox
        for bb in labels:
            with open(os.path.join(LABEL_INTER_PATH, '{}_{:06d}.txt'.format(path[-8:-4], bb[0])), 'a') as file:
                if bb[5] <= 0:
                    continue
                # limit interested classed
                if bb[1] in ['Cyclist', 'Van', 'Car', 'Misc', 'Pedestrian', 'DontCare', 'Person', 'Person_sitting']:
                    continue
                # merge truck and tram
                if bb[1] == 'Tram':
                    bb[1] = 'Truck'

                # 读取图像大小
                img_size = Image.open(os.path.join(
                    root.format('image_02'),
                    '{}/{:06d}.png'.format(path[-8:-4], bb[0]))).size
                img_size = (img_size[0], img_size[1] - CROP_SIZE)  # 宽度减小CROP_SIZE
                # 重写bb
                # center_x = (bb[2] + bb[4]) / 2 / img_size[0]
                # center_y = (bb[3] + bb[5]) / 2 / img_size[1]
                # bb_width = abs((bb[2] - bb[4])) / img_size[0]
                # bb_height = abs((bb[3] - bb[5])) / img_size[1]
                top_x = bb[2] / img_size[0] * TARGET_IMG_SIZE[0]
                top_y = bb[3] / img_size[1] * TARGET_IMG_SIZE[1]
                bottom_x = bb[4] / img_size[0] * TARGET_IMG_SIZE[0]
                bottom_y = bb[5] / img_size[1] * TARGET_IMG_SIZE[1]
                file.write('{} {:.7f} {:.7f} {:.7f} {:.7f}\n'.format(bb[1], top_x, top_y, bottom_x, bottom_y))
                print(path)
                # record categories
                categories.add(bb[1])
                categories_dist[bb[1]] += 1
                print(categories, categories_dist)
            if vis:
                # 可视化
                syn_img = cv2.imread(os.path.join(
                    '/home/hviktortsoi/Code/pix2pixHD/results/OC_PC2head_ResPrevFusion_sBatch_DilationConv/test_latest_local_tracking_numD_2/images',
                    '{}_{:06d}_synthesized_image.png'.format(path[-8:-4], bb[0])))
                if syn_img is None: continue
                syn_img = cv2.resize(syn_img, img_size)
                # img = img[120:, ...]
                cv2.rectangle(syn_img, tuple(np.int_(bb[2:4])), tuple(np.int_(bb[4:6])), (0, 255, 0), 4)

                real_img = cv2.imread(os.path.join(
                    '/media/hvt/95f846d8-d39c-4a04-8b28-030feb1957c6/dataset/KITTI/tracking/training/image_02',
                    '{}/{:06d}.png'.format(path[-8:-4], bb[0])))
                real_img = real_img[120:, ...]

                img = np.concatenate([syn_img, real_img], axis=0)
                cv2.imshow('', img)
                cv2.waitKey(0)
                print(bb)




def parse_label(path, img_id, width, height, task):
    """
    convert KITTI label to coco format
    :param path:
    :param img_id:
    :param width:
    :param height:
    :return: annotations
    """
    try:
        labels = pd.read_csv(path, delimiter=' ', header=None).values
    except Exception:
        print('Empty Label: {}'.format(path))
        return []
    annotations = []
    # 找到此张图像上对应的bb
    for idx, label in enumerate(labels):
        category_id = label[0] if task == 'instance' else (fixed_category[label[0]] + 1)
        print("id: category: ", category_id, label)
        top_x, top_y, bottom_x, bottom_y = label[4:8]
        area = label[5] if task == 'instance' else (bottom_x - top_x) * (bottom_y - top_y)
        annotations.append({
            'id': '{}_{}'.format(img_id, idx),
            'image_id': img_id,
            'category_id': category_id,
            'segmentation': {'counts': label[-1], 'size': tuple(label[6:8])} if task == 'instance' else None,
            'area': area,
            'bbox': [top_x, top_y, bottom_x - top_x, bottom_y - top_y],
            'iscrowd': 0,
        })
        categories.add(category_id)
    return annotations


def process_dataset(img_list, name, task):
    images = []
    annotations = []

    for idx, img_name in enumerate(sorted(img_list)):
        img_path = os.path.join(img_root, img_name)
        height, width = cv2.imread(img_path).shape[:2]
        img_name_wo_suffix = img_name[:img_name.rfind('.')].replace('/', '_')
        img_id = int(img_name_wo_suffix) if dataset_type == 'object' else int(img_name_wo_suffix.replace('_', ''))
        label_path = os.path.join(LABEL_INTER_PATH, '{}.txt'.format(img_name_wo_suffix))

        # 解析标签
        annotation = parse_label(label_path, img_id=img_id, width=width, height=height, task=task)

        images.append({
            'license': 3,
            'file_name': img_name.replace('/', '_'),
            'width': TARGET_IMG_SIZE[0],
            'height': TARGET_IMG_SIZE[1],
            'id': img_id,
            'coco_url': '', 'date_captured': '', 'flickr_url': '',
        })
        annotations.extend(annotation)

        print('{}/{} {}'.format(idx, len(img_list), img_name))
        print(categories)

    dataset = {
        'info': {
            'description': 'KITTI Synthesis',
            'url': '',
            'version': '0.1',
            'year': 2020,
            'contributor': 'HVT@BDBC',
            'date_created': '2020/01/11'
        },
        'images': images,
        'annotations': annotations,
        "categories": [{
            "id": int(category_id),
            "name": str(category_id),
            "supercategory": 'str',
        } for category_id in categories]

    }

    json.dump(dataset, open(
        os.path.join(target_path, '{}_KITTI_{}_{}.json')
            .format(task, dataset_type, name), 'w'), indent=0)


if __name__ == '__main__':
    # KITTI_tracking_to_intermediate()
    # exit()

    # dataset_type = 'synthesised'
    dataset_type = 'raw'
    # dataset_type = 'tracking'
    # assert dataset_type in ['tracking', 'object', 'synthesised']

    dataset_root = '../../resources/Kitti/{}/training'.format(dataset_type)
    img_root = "D:\Programming\PycharmProjects\\resources\Kitti\Kitti\\raw\\training\\image_2"
    target_path = 'D:\Programming\PycharmProjects\\resources\Kitti/transform'
    # img_root = '/home/hviktortsoi/Code/pix2pixHD/results/OC_PC2head_ResPrevFusion_sBatch_DilationConv/test_latest_local_tracking_numD_2/images'
    # target_path = '/home/hviktortsoi/Code/pix2pixHD/results/OC_PC2head_ResPrevFusion_sBatch_DilationConv/test_latest_local_tracking_numD_2'

    if dataset_type == 'tracking' or dataset_type == 'raw':
        img_list = os.listdir('D:\Programming\PycharmProjects\\resources\Kitti\Kitti\\raw\\training\\image_2')
        # img_list = list(itertools.chain(*img_list))
    elif dataset_type == 'object':
        img_list = os.listdir(img_root)
    elif dataset_type == 'synthesised':
        img_list = [img for img in os.listdir(img_root)]
    else:
        raise NotImplementedError()

    # # crop real image
    # rescale_real_img(img_list)
    # exit()
    # process_dataset(img_list, 'real_full', task='instance')
    # process_dataset(img_list, 'val_full', task='instance')
    # exit()

    # 划分训练测试集
    random.shuffle(img_list)
    split = int(len(img_list) * 0.8)
    train_list = img_list[:split]
    val_list = img_list[split:]

    # process_dataset(train_list, 'train', task='instance')
    # process_dataset(val_list, 'val', task='instance')
    process_dataset(train_list, 'train', task='detection')
    process_dataset(val_list, 'val', task='detection')
