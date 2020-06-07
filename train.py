"""
Data parameters
---------------

data_folder: folder with data files
models_folder: folder to save the trained models
keep_difficult: use objects considered difficult to detect?

Model parameters
----------------

Not too many here since the SSD300 has a very specific structure
number of different types of objects

checkpoint: path to model checkpoint, None if none
batch_size: batch size
iterations: number of iterations to train
workers number of workers for loading data in the DataLoader
print_freq: print training status every __ batches
lr: learning rate
decay_lr_at: decay learning rate after these many iterations
decay_lr_to: decay learning rate to this fraction of the existing learning rate
momentum: momentum
weight_decay: weight decay
grad_clip: clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculationfrom pathlib import Path
"""

from pathlib import Path
import argparse

from objectDetection import main_train
from objectDetection.utils import label_map
from objectDetection.jutils import print_params

ROOT_DIR = Path('./')

params = {
        'ROOT_DIR': ROOT_DIR,
        'data_folder' : ROOT_DIR/'output',
        'models_folder' : ROOT_DIR/'output',
        'keep_difficult' : True,
        'n_classes' : len(label_map),
        'checkpoint' : None,
        'batch_size' : 8,
        'iterations' : 120000,
        'workers' : 4,
        'print_freq' : 200,
        'lr' : 1e-3,
        'decay_lr_at' : [80000, 100000],
        'decay_lr_to' : 0.1,
        'momentum' : 0.9,
        'weight_decay' : 5e-4,
        'grad_clip': None,
    }

# cudnn.benchmark = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder', type=str, default='./data',
                        help="The folder that stores the data to load")
    parser.add_argument('models_folder', type=str, default='./models',
                        help="The dir to save the trained models")
    parser.add_argument('--keep_difficult', type=bool, default=True,
                        help="Whether difficult examples must be used")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help="The path of the checkpoint file if we are resuming training")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="The batch size for training")
    parser.add_argument('--iterations', type=int, default=120000,
                        help="Number of epochs to run")

    args = parser.parse_args()

    params['data_folder'] = args.data_folder
    params['models_folder'] = args.models_folder
    params['keep_difficult'] = args.keep_difficult
    params['checkpoint'] = args.checkpoint
    params['batch_size'] = args.batch_size
    params['iterations'] = args.iterations

    print_params(params)
    main_train(params)
