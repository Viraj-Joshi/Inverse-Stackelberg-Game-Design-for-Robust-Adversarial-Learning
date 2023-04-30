import os
from unittest.mock import patch
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
import sys
import random
import torch.utils.tensorboard as tb

from torch.utils.data import DataLoader, Dataset
from skimage.io import imread
from PIL import Image
from os import path

from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    OpenGLPerspectiveCameras,
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    HardPhongShader,
    TexturesUV,
    BlendParams,
    SoftSilhouetteShader
)

from MeshDataset import MeshDataset
from BackgroundDataset import BackgroundDataset
from darknet import Darknet
from loss import TotalVariation, dis_loss, calc_acc, TotalVariation_3d
from generator import Generator
from learner import Learner

from torchvision.utils import save_image
import torchvision
import random

from PIL import ImageDraw
from faster_rcnn.dataset.base import Base as DatasetBase
from faster_rcnn.backbone.base import Base as BackboneBase
from faster_rcnn.bbox import BBox
from faster_rcnn.model import Model as FasterRCNN
from faster_rcnn.roi.pooler import Pooler
from faster_rcnn.config.eval_config import EvalConfig as Config

def main():
    torch.cuda.empty_cache()
    import argparse

    parser = argparse.ArgumentParser()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    parser.add_argument('--img_dir', type=str, default='inria/Train/pos')
    parser.add_argument('--lab_dir', type=str, default='inria/Train/pos/yolo-labels')
    parser.add_argument('--mesh_dir', type=str, default='data/meshes')
    parser.add_argument('--bg_dir', type=str, default='data/background')
    parser.add_argument('--test_bg_dir', type=str, default='data/test_background')
    parser.add_argument('--output', type=str, default='out/patch')

    parser.add_argument('--patch_num', type=int, default=1)
    parser.add_argument('--patch_dir', type=str, default='')
    parser.add_argument('--idx', type=str, default='idx/chest_legs1.idx')
    
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--learner_epochs', type=int, default=1)
    parser.add_argument('--generator_epochs', type=int, default=1) #20 

    parser.add_argument('--img_size', type=int, default=416)
    parser.add_argument('--num_bgs', type=int, default=10)
    parser.add_argument('--num_test_bgs', type=int, default=2)
    parser.add_argument('--num_angles_test', type=int, default=1)
    parser.add_argument('--angle_range_test', type=int, default=0)
    parser.add_argument('--num_angles_train', type=int, default=1)
    parser.add_argument('--angle_range_train', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--rand_translation', type=int, default=50)
    parser.add_argument('--num_meshes', type=int, default=1)

    parser.add_argument('--cfgfile', type=str, default="cfg/yolo.cfg")
    parser.add_argument('--weightfile', type=str, default="data/yolov2/yolov2_20000.weights")
    parser.add_argument('--generator_weightfile', type=str, default="")
    
    parser.add_argument('--detector', type=str, default='yolov2')
    parser.add_argument('--test_only', action='store_true')

    parser.add_argument('--log_dir', type=str, default='.')


    config = parser.parse_args()
    learner = Learner(config, device)
    generator = Generator(config, device)

    if config.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(config.log_dir, 'train'), flush_secs=1)

    
    exp_relay = []
    adv_patch = generator.initialize_patch()
    adv_patch.requires_grad_(True)
    for iteration in range(config.iterations+1):
        print("#############################iteration:",iteration)
        ########
        # Follower's TURN
        ########

        adv_images = generator.update(learner,adv_patch,train_logger,iteration,exp_relay) # update the generator's parameters based on leader's score 
        
        ###########
        # Learner's TURN
        ###########
        learner.update(adv_images,iteration,train_logger)
    train_logger.close()

if __name__ == '__main__':
    main()

# python train.py --mesh_dir=data/meshes --iterations=200 --generator_epochs=10  --num_bgs=32 --num_test_bgs=512 --batch_size=16 --num_angles_train=1 --angle_range_train=0 --num_angles_test=1 --angle_range_test=0 --idx=idx/chest_legs1.idx --detector=yolov2 --patch_dir=example_logos/fasterrcnn_chest_legs
