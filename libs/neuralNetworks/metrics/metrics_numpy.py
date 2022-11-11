
import numpy as np
import torch
# from numba import njit
import cupy as cp
import gc
from math import sqrt
from sklearn.metrics import mean_squared_error


SMOOTH = 1e-6

# combining to on function so as  to improve performances
# @njit


def get_confusion_matrix(outputs, labels, threshold=0.5, use_cupy=False):
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.cpu().numpy() # inference mode, do not need detach()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    if use_cupy:
        labels = cp.asarray(labels)
        outputs= cp.asarray(outputs, cp.float16)

        gt_positive = (labels >= threshold)
        gt_negative = (labels < threshold)
        pred_positive = (outputs >= threshold)
        pred_negative = (outputs < threshold)

        TP = (gt_positive) & (pred_positive)
        TP_num = cp.sum(TP)
        TN = (gt_negative) & (pred_negative)
        TN_num = cp.sum(TN)
        FP = (gt_negative) & (pred_positive)
        FP_num = cp.sum(FP)
        FN = (gt_positive) & (pred_negative)
        FN_num = cp.sum(FN)

        # The gpu memory is limited
        del gt_positive
        del gt_negative
        del pred_positive
        del pred_negative
        del TP
        del TN
        del FP
        del FN
        gc.collect()
        torch.cuda.empty_cache()

        TP_num = cp.asnumpy(TP_num)
        TN_num = cp.asnumpy(TN_num)
        FP_num = cp.asnumpy(FP_num)
        FN_num = cp.asnumpy(FN_num)
        gc.collect()
        torch.cuda.empty_cache()

    else:
        gt_positive = (labels >= threshold)
        gt_negative = (labels < threshold)
        pred_positive = (outputs >= threshold)
        pred_negative = (outputs < threshold)

        TP = (gt_positive) & (pred_positive)
        TP_num = np.sum(TP)
        TN = (gt_negative) & (pred_negative)
        TN_num = np.sum(TN)
        FP = (gt_negative) & (pred_positive)
        FP_num = np.sum(FP)
        FN = (gt_positive) & (pred_negative)
        FN_num = np.sum(FN)

    return TP_num, TN_num, FP_num, FN_num


def get_metrics(TP_num, TN_num, FP_num, FN_num):

    ACC = (TP_num+TN_num) / (TP_num+TN_num+FP_num+FN_num)

    if TP_num + FN_num > 0:
        TPR = TP_num / (TP_num + FN_num)  #sensitivity, recall
    else:
        TPR = 1

    if FP_num + TN_num > 0:
        TNR = TN_num / (FP_num + TN_num)  #specificity
    else:
        TNR = 1

    # if TP_num + FP_num > 0:
    #     PPV = TP_num / (TP_num + FP_num)   #positive predictive value, precision
    # else:
    #     PPV = 1
    #
    # if TN_num + FN_num > 0:
    #     NPV = TN_num / (TN_num + FN_num)   #negative true predictive value
    # else:
    #     NPV = 1


    # if TP_num + FP_num + FN_num > 0:
    #     IOU = TP_num / (TP_num + FP_num + FN_num)
    # else:
    #     IOU = 1

    if 2 * TP_num + FP_num + FN_num > 0:
        DICE = (2 * TP_num) / (2 * TP_num + FP_num + FN_num)
    else:
        DICE = 1

    return ACC, TPR, TNR, DICE

#or from sklearn.metrics import fbeta_score
def get_f_score(TP_num, FP_num, FN_num, b=1):

    if TP_num + FN_num > 0:
        TPR = TP_num / (TP_num + FN_num)  #sensitivity, recall
    else:
        TPR = 1

    if TP_num + FP_num > 0:
        PPV = TP_num / (TP_num + FP_num)   #positive predictive value, precision
    else:
        PPV = 1

    precision = TPR
    recall = PPV

    if precision + recall == 0:
        f_score = 0
    else:
        # f_score = 2 * precision * recall / (precision + recall)
        f_score = (1 + b**2) / (b**2/recall + 1/precision)

    return f_score


def get_acc(outputs, labels, threshold=0.5):
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

    TP = (labels > threshold) & (outputs > threshold)
    TP_num = np.sum(TP)
    TN = (labels < threshold) & (outputs < threshold)
    TN_num = np.sum(TN)
    FP = (labels < threshold) & (outputs > threshold)
    FP_num = np.sum(FP)
    FN = (labels > threshold) & (outputs < threshold)
    FN_num = np.sum(FN)

    acc = (TP_num+TN_num) / (TP_num+TN_num+FP_num+FN_num)

    return acc


def get_sen(outputs, labels, threshold=0.5):
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

    TP = (labels > threshold) & (outputs > threshold)
    TP_num = np.sum(TP)
    FN = (labels > threshold) & (outputs < threshold)
    FN_num = np.sum(FN)

    if TP_num + FN_num > 0:
        TPR = TP_num / (TP_num + FN_num)
    else:
        TPR = 1

    return TPR


def get_spe(outputs, labels, threshold=0.5):
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

    TN = (labels < threshold) & (outputs < threshold)
    TN_num = np.sum(TN)
    FP = (labels < threshold) & (outputs > threshold)
    FP_num = np.sum(FP)

    if FP_num + TN_num > 0:
        TNR = TN_num / (FP_num + TN_num)
    else:
        TNR = 1

    return TNR


def cal_rmse(predictions, targets):
    return sqrt(mean_squared_error(predictions, targets))


# IoU = TP / (TP+FP+FN)
# Jaccard = |A∩B| / |A∪B| = TP / (TP + FP + FN)
def get_iou(outputs, labels, threshold=0.5):
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    #（N,C,H,W) -> (N,H,W)
    outputs = outputs.squeeze(1) > threshold
    labels = labels.squeeze(1).astype(np.int) > threshold

    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    return iou.mean()



# Dice = 2 |A∩B| / (|A|+|B|) = 2 TP / (2 TP + FP + FN)
# Dice = (TP+TP) / (TP+TP+FP+FN)
# Dice = 2 * IoU / (IoU + 1)
def get_dice(outputs, labels, threshold=0.5):
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    #（N,C,H,W) -> (N,H,W)
    outputs = outputs.squeeze(1) > threshold
    labels = labels.squeeze(1).astype(np.int) > threshold

    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (2 * intersection + SMOOTH) / (union +intersection + SMOOTH)

    return iou.mean()
