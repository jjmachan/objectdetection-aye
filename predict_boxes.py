import argparse

import torch
from torchvision import transforms
from PIL import Image

from objectDetection.model import SSD300
from objectDetection.definitions import ROOT_DIR,device
from objectDetection.utils import label_map, rev_label_map
from objectDetection import InferencingModel

def main(checkpoint_path, image_path):
    model = InferencingModel(checkpoint_file=checkpoint_path)

    # load img
    orginal_img = Image.open(image_path, mode='r')
    orginal_img = orginal_img.convert('RGB')

    labels, boxes = model(orginal_img, min_score=0.2, top_k=200, max_overlap=0.5)

    print(labels, boxes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', default='./output/checkpoint.pth.tar',
                        help='Path of the checkpoint to load')
    parser.add_argument('image',
                        help='Path of the image to predict the bounding boxes')
    args = parser.parse_args()

    main(args.checkpoint, args.image)
