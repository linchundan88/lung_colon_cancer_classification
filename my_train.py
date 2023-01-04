'''The training file, this code can be invoked by my_train.sh through the command line.
the program should under the scope of if __name__ == '__main__',
  otherwise the load model and other operations will be executed multiple times because of multi processors.
'''

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))  #print(sys.path)
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_devices', default='0,1')  # '0,1'
parser.add_argument('--task_type', default='LC25000_5_classes_cv0')  #LC25000  LC25000_5_classes, cross validation times:0
parser.add_argument('--model_name', default='densenet121')
parser.add_argument('--smooth_factor', type=float, default='0.0')   #label smoothing
# parser.add_argument('--class_weights', nargs="+", type=float, default=(1, 3, 3))
parser.add_argument('--use_amp', action='store_true', default=True)  # AUTOMATIC MIXED PRECISION
parser.add_argument('--epochs_num', type=int, default=20)
parser.add_argument('--weight_decay', type=float, default=0)  #L2 regularization, 1e-4 is too big
parser.add_argument('--lr', type=float, default=0.0001)  # 0.001 or 0.0001
# parser.add_argument('--step_size', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=64)
#recommend num_workers = the number of gpus * 4, when debugging it should be set to 0.
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--torch_script', action='store_true')  # before saving a model, using torch script JIT to compile the model.
parser.add_argument('--save_only_state_dict', default=True)
parser.add_argument('--path_save', default='/tmp2')
parser.add_argument('--parallel_mode', choices=['DP', 'DDP'], default='DP')   # DP:Data Parallel,  DDP:Distributed Data Data Parallel
parser.add_argument('--sync_batchnorm', default=True)   #for Distributed Data Parallel training
args = parser.parse_args()
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices   # setting  GPUs, must before import torch
from libs.dataset.my_dataset import Dataset_multiclass
import torch
import torch.nn as nn
import torch.optim as optim
# import torch_optimizer as optim_plus
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, CyclicLR, ConstantLR, CosineAnnealingLR
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from libs.neuralNetworks.utils.my_model import create_model, get_model_shape
from libs.neuralNetworks.utils.my_train import train_DP, train_DDP, draw_loss_graph
import torch.multiprocessing as mp
from munch import Munch
from libs.dataset.my_data_dist import get_class_distribution_mc


if __name__ == '__main__':
    path_csv = Path(__file__).resolve().parent / 'datafiles'
    csv_train = path_csv / f'{args.task_type}_train.csv'
    csv_valid = path_csv / f'{args.task_type}_valid.csv'

    num_classes, _, _ = get_class_distribution_mc(csv_train)
    image_shape = get_model_shape(args.model_name)
    # model = create_model(args.model_name, num_classes=num_classes)
    model = create_model(args.model_name, num_classes=num_classes)

    transform_train = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # A.Blur(blur_limit=3, p=0.5),
        # A.RandomCrop(width=args.image_shape[1], height=args.image_shape[0]),
        A.RandomRotate90(p=0.8),
        A.Resize(image_shape[0], image_shape[1]),
        # A.ShiftScaleRotate(p=0.1, shift_limit=0.05, scale_limit=0.1, rotate_limit=10),
        # A.Affine(scale=0.1, rotate=10, translate_percent=0.1),
        A.RandomBrightnessContrast(p=0.8, brightness_limit=0.1, contrast_limit=0.1),
        A.GaussianBlur(p=0.6, blur_limit=(3, 5), sigma_limit=0),
        ToTensorV2()
    ])

    ds_train = Dataset_multiclass(csv_file=csv_train, transform=transform_train, image_shape=image_shape)
    ds_valid = Dataset_multiclass(csv_file=csv_valid, image_shape=image_shape)

    # class_weights = torch.FloatTensor(args.class_weights)
    # if torch.cuda.device_count() > 0:
    #     class_weights = class_weights.cuda()  #for DDP, class_weights should be moved to the corresponding gpu.
    # criterion = nn.CrossEntropyLoss(reduction='mean', class_weights=class_weights, label_smoothing=args.smooth_factor)  # Softmax + Cross-Entropy Loss, LogSoftmax + NLLLoss
    criterion = nn.CrossEntropyLoss(reduction='mean', label_smoothing=args.smooth_factor)  # Softmax + Cross-Entropy Loss, LogSoftmax + NLLLoss
    if args.parallel_mode == 'DDP':
        n_gpus = torch.cuda.device_count()
        args.lr *= n_gpus
        args.batch_size = int(args.batch_size / n_gpus)  # total batch size,  batch size per gpu
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  #lr 1e-3 for the CosineAnnealingLR scheduler
    # optimizer = optim_plus.radam(model.parameters(), lr=0.001, weight_decay=args.weight_decay)
    # optimizer = optim_plus.Lookahead(optimizer, k=5, alpha=0.5)

    # scheduler = CosineAnnealingLR(optimizer, T_max=3)
    scheduler = ConstantLR(optimizer, factor=0.1, total_iters=1)  # warm up using a small lr, and then setting a stable lr.  if factor=1,do not change lr,
    # scheduler = ExponentialLR(optimizer, gamma=args.gamma)  #lr:1e-4, gamma:0.6
    # epochs_num = args.epochs_num
    # scheduler = MultiStepLR(optimizer, milestones = [int(epochs_num * 0.3), int(epochs_num * 0.6), int(epochs_num * 0.9)], gamma=0.1)
    #Big Transfer (BiT): General Visual Representation Learning
    #During fine-tuning, we decay the learning rate by a factor of 10 at 30%, 60% and 90% of the training steps
    # scheduler = CosineAnnealingLR(optimizer, T_max= 4, eta_min=0)  #T_max: half of one circle
    # scheduler = GradualWarmupScheduler(optimizer, multiplier=4, total_epoch=4, after_scheduler=scheduler)
    # scheduler = CyclicLR(optimizer, 1e-5, 1e-3, step_size_up=100, mode='triangular')
    if isinstance(scheduler, CyclicLR):
        scheduler_mode = 'batch'
    else:
        scheduler_mode = 'epoch'

    path_save = Path(args.path_save)

    # Munch is better than Dict. The contents of it can be accessed by dot operator or string name.
    train_config = Munch({
        'model': model, 'ds_train': ds_train, 'list_ds_valid': [ds_valid],
        'criterion': criterion, 'optimizer': optimizer, 'scheduler': scheduler, 'scheduler_mode': scheduler_mode,
        'use_amp': args.use_amp, 'epochs_num': args.epochs_num,
        'batch_size': args.batch_size, 'num_workers': args.num_workers,
        'torch_script': args.torch_script, 'save_only_state_dict': args.save_only_state_dict, 'path_save': path_save,
        'losses_pkl': path_save / 'losses.pkl'
    })

    if args.parallel_mode == 'DP':
        train_DP(train_config)
    elif args.parallel_mode == 'DDP':
        train_config.sync_batchnorm = args.sync_batchnorm
        if args.model_name in ['inception_resnet_v2']:  # only for inception_resnet_v2, otherwise runtime error MapAllocator.cpp":263,
            torch.multiprocessing.set_sharing_strategy('file_system')
        mp.spawn(train_DDP, args=(n_gpus, train_config), nprocs=n_gpus, join=True)

    # ddp training can not return list_losses under multiple processors, so use pickle.
    import pickle
    list_losses = pickle.load(open(train_config.losses_pkl, 'rb'))  # the multi processer function can not return values
    draw_loss_graph(list_losses, path_save / 'losses.png')

    print('OK')





