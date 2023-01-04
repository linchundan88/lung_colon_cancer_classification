'''
    train_DDP: distributed data parallel。 It is invoked by mp.spawn in my_train.py.
    my_train_DP: data parallel
    train_one_epoch: used by both DP and DDP training.
    validate: used by both DP and DDP training.
    draw_loss_graph:


'''

import warnings
warnings.filterwarnings("ignore")
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from warmup_scheduler import GradualWarmupScheduler
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import logging
from datetime import datetime
import pickle
import gc



#region distributed data parallel training
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12354'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def train_DDP(rank, world_size, config):
    model1 = config.model
    ds_train = config.ds_train
    sync_batchnorm = config.sync_batchnorm
    list_ds_valid = config.list_ds_valid
    optimizer = config.optimizer
    criterion = config.criterion
    scheduler = config.scheduler
    scheduler_mode = config.scheduler_mode
    use_amp = config.use_amp
    epochs_num = config.epochs_num
    batch_size = config.batch_size
    num_workers = config.num_workers
    torch_script = config.torch_script
    save_only_state_dict = config.save_only_state_dict
    path_save = config.path_save
    losses_pkl = config.losses_pkl

    setup(rank, world_size)

    if rank == 0:
        path_save.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=path_save / f'train{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.log', level=logging.DEBUG)
        list_loss_history = []  #using list_loss_history to store losses of all epoches.  It is used to draw the line of training and validation loss

    torch.cuda.set_device(rank)  #dist.all_gather_object runs much quicker.
    if sync_batchnorm:
        model1 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model1)  # it will slow down the training process.
    model1.to(rank)
    model_ddp = DDP(model1, device_ids=[rank])

    #from torch.utils.data import Subset
    #in the future, dynamic resampling, manually divide dataset across multi processors
    # train_subset = Subset(train_set, subset_indices)
    sampler_train = DistributedSampler(ds_train, num_replicas=world_size, rank=rank)
    loader_train = DataLoader(ds_train, batch_size=batch_size, sampler=sampler_train, num_workers=num_workers, pin_memory=True)

    for epoch in range(epochs_num):
        if rank == 0:
            # logging.info(f'training epoch {epoch}/{epochs_num - 1} at:' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
            print(f'training epoch {epoch}/{epochs_num - 1} at:' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        sampler_train.set_epoch(epoch)
        loss_train = _train_one_epoch(epoch, model_ddp, loader_train, criterion, optimizer, scheduler, scheduler_mode, use_amp, rank)
        dist.barrier()
        data = {
            'loss': loss_train
        }
        outputs = [None for _ in range(world_size)]
        dist.all_gather_object(outputs, data)

        if rank == 0:
            list_loss = [output['loss'] for output in outputs]
            loss_train_avg = float(np.mean(list_loss))  #combine loss from multiple processes.  numpy->float in order to support pickle
            print(f'epoch:{epoch}  training loss:{loss_train_avg:.4f}')
            list_epoch_losses = [loss_train_avg]  # add training loss

        for index, ds_valid in enumerate(list_ds_valid):
            if rank == 0:
                print(f'computing validation dataset {index}...')
            sampler_valid= DistributedSampler(ds_valid, world_size, rank)
            dataloader_valid = DataLoader(ds_valid, batch_size=batch_size, sampler=sampler_valid, num_workers=num_workers, pin_memory=True)
            loss_valid, gt_labels, pred_labels = _validate_one_epoch(model_ddp, dataloader_valid, criterion, use_amp=use_amp, rank=rank)
            dist.barrier()
            data = {
                'loss': loss_valid, 'gt_labels': gt_labels, 'pred_labels': pred_labels
            }
            outputs = [None for _ in range(world_size)]
            dist.all_gather_object(outputs, data)

            if rank == 0:
                list_loss = [output['loss'] for output in outputs]
                list_gt_labels = [output['gt_labels'] for output in outputs]
                list_pred_labels = [output['pred_labels'] for output in outputs]

                loss_valid_avg = float(np.mean(list_loss))  # average over multiple processers,  numpy->float in order to support pickle
                list_epoch_losses.append(loss_valid_avg)  # list_epoch_losses:[loss_train, loss_validation1, loss_validation2...] for every epoch
                gt_labels = np.hstack(list_gt_labels).tolist()
                pred_labels = np.hstack(list_pred_labels).tolist()

                print(f'epoch:{epoch} validation dataset {index} loss:{loss_valid_avg:.4f}')
                print(confusion_matrix(gt_labels, pred_labels))

        if rank == 0:
            list_loss_history.append(list_epoch_losses)
            save_model_file = path_save / f'valid_loss_{loss_valid_avg:.4f}_epoch{epoch}.pth'  # the loss of the last validation dataset
            print('save model:', save_model_file)
            if not torch_script:
                if save_only_state_dict:
                    torch.save(model_ddp.module.state_dict(), save_model_file)
                else:
                    torch.save(model_ddp.module, save_model_file)
            else:
                script_model = torch.jit.script(model_ddp.module)
                script_model.save(model_ddp.module)

    dist.barrier()
    cleanup()

    if rank == 0:
        pickle.dump(list_loss_history, open(losses_pkl, 'wb'))

#endregion


def train_DP(config):
    model1 = config.model
    ds_train = config.ds_train
    list_ds_valid = config.list_ds_valid
    optimizer = config.optimizer
    criterion = config.criterion
    scheduler = config.scheduler
    scheduler_mode = config.scheduler_mode
    use_amp = config.use_amp
    epochs_num = config.epochs_num
    batch_size = config.batch_size
    num_workers = config.num_workers
    torch_script = config.torch_script
    save_only_state_dict = config.save_only_state_dict
    path_save = config.path_save
    losses_pkl = config.losses_pkl

    path_save.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=path_save / f'train{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.log', level=logging.DEBUG)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0:
        model1.to(device)
    if torch.cuda.device_count() > 1:
        model1 = nn.DataParallel(model1)  # using os.environ["CUDA_VISIBLE_DEVICES"] instead of device_ids=[0,1,2]

    list_loss_history = []  #using list_loss_history to draw the line of training and validation loss
    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    for epoch in range(epochs_num):
        logging.info(f'training epoch {epoch}/{epochs_num - 1} at:' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        loss_train = _train_one_epoch(epoch, model1, loader_train, criterion, optimizer, scheduler, scheduler_mode, use_amp)
        print(f'epoch:{epoch} training losses:{loss_train:.4f}')
        list_epoch_losses = [float(loss_train)]  # add training loss, numpy float64 -> float, because pickle

        for index, ds_valid in enumerate(list_ds_valid):  # can have multiple validation datasets
            print(f'computing validation dataset {index}')
            dataloader_valid = DataLoader(ds_valid, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
            loss_valid, gt_labels, pred_labels = _validate_one_epoch(model1, dataloader_valid, criterion, use_amp=use_amp)
            print(f'epoch:{epoch} validation dataset {index} loss:{loss_valid:.4f}')
            list_epoch_losses.append(float(loss_valid))   # add validation loss, numpy float64 -> float, because pickle
            print(confusion_matrix(gt_labels, pred_labels))

        list_loss_history.append(list_epoch_losses)  #list_epoch_losses: []train_loss, validation1_loss, validation2_loss...]

        save_model_file = path_save / f'valid_loss_{loss_valid:.4f}_epoch{epoch}.pth'
        print('save model:', save_model_file)
        if isinstance(model1, nn.DataParallel):  # or hasattr(module1, 'module')
            model1 = model1.module  #model1 is a wrapper in data parallel model, get the actual model.
        if not torch_script:
            if save_only_state_dict:
                torch.save(model1.state_dict(), save_model_file)
            else:
                torch.save(model1, save_model_file)
        else:
            script_model = torch.jit.script(model1)
            script_model.save(save_model_file)

    pickle.dump(list_loss_history, open(losses_pkl, 'wb'))



def _train_one_epoch(epoch, model, dataloader, criterion, optimizer, scheduler, scheduler_mode, use_amp, rank=None):

    if rank is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cuda', rank)  # distributed data parallel
    model.train()

    list_loss = []
    # iters = len(dataloader)   datasize / batch_size
    scaler = GradScaler(enabled=use_amp)

    if rank is None or rank==0:
        dataloader = tqdm(dataloader, desc=f'Training epoch {epoch}')  # the processor other than rank0 do not show progressbar.
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # automatic mixed precision https://pytorch.org/docs/stable/notes/amp_examples.html
        # https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
        # outputs = model(inputs)
        # loss = criterion(outputs, labels)
        # loss.backward()
        # optimizer.step()
        # optimizer.zero_grad()

        with autocast(enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler_mode == 'batch':
            scheduler.step()

        list_loss.append(loss.item())  # The item() method extracts the loss’s value as a Python float.
        # logging.info(f'training batch:{batch_idx}, losses_obsoleted:{loss.item():.2f}')

    if scheduler_mode == 'epoch':
        scheduler.step()

    clear_gpu_cache()

    return np.mean(list_loss)


@torch.inference_mode()
def _validate_one_epoch(model, dataloader, criterion, use_amp, rank=None):
    if rank is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cuda', rank)  # distributed data parallel
    model.eval()
    list_losses, list_gt_labels, list_pred_labels = [], [], []

    if rank is None or rank == 0:
        dataloader = tqdm(dataloader, desc=f'Validation')  # the processor other than rank0 do not show progressbar.
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            outputs = F.softmax(outputs, dim=1)
            pred_labels = torch.argmax(outputs, dim=1)

        list_losses.append(loss.item())
        list_gt_labels.append(labels.cpu().numpy())
        list_pred_labels.append(pred_labels.cpu().numpy())

    clear_gpu_cache()

    loss_mean = np.mean(list_losses)
    gt_labels = np.hstack(list_gt_labels)  #one dimension
    pred_labels = np.hstack(list_pred_labels)
    # pred_labels = np.argmax(np.vstack(list_preds), axis=-1)  using pytorch do argmax

    return loss_mean, gt_labels, pred_labels


def clear_gpu_cache():
    if torch.cuda.device_count() > 0:
        gc.collect()
        torch.cuda.empty_cache()


def draw_loss_graph(list_losses, save_img_file=None):
    import matplotlib.pyplot as plt

    train_losses = [loss[0] for loss in list_losses]
    val_losses = [loss[1] for loss in list_losses]

    plt.figure(figsize=(10, 5))
    plt.title("Training and Validation Loss")
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    if save_img_file is not None:
        plt.savefig(save_img_file, bbox_inches='tight') #save image file should be executed before calling plt.show()
    else:
        plt.show()
        # plt.close()



