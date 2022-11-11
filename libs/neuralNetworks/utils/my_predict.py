'''
  predict_single_model:
      using data parallel. It is similiar to the validate_one_epoch function in my_train.py, except for DDP support, loss, and argmax.
      In the future, I want to change it to distributed data parallel inference and using torch script JIT to boost performance.
  predict_multi_models: do weighted average
  predict_one_image: for an image, output its predicted probabilities and class.

'''

import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import autocast
from tqdm import tqdm
from libs.neuralNetworks.utils.my_img_to_tensor import img_to_tensor
from libs.neuralNetworks.utils.my_train import clear_gpu_cache


@torch.inference_mode()  # or with torch.inference_mode():
def predict_single_model(model, data_loader, use_amp):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    if torch.cuda.device_count() > 0:
        model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    list_probs = []
    for batch_idx, inputs in enumerate(tqdm(data_loader)):
        if isinstance(inputs, list):  # both images and labels
            inputs = inputs[0]
        inputs = inputs.to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)
        list_probs.append(probabilities.cpu().numpy())
    probs = np.vstack(list_probs)

    clear_gpu_cache()

    y_preds = probs.argmax(axis=-1)
    return probs, y_preds



#different models may use different image sizes, so need different data_loaders
def predict_multi_models(model_dicts):
    list_probs, list_y_preds = [], []

    for model_dict in model_dicts:
        data_loader = model_dict['data_loader']
        probs, y_preds = predict_single_model(model_dict['model'], data_loader, model_dict['use_amp'])
        list_probs.append(probs)
        list_y_preds.append(y_preds)
        # del model
        clear_gpu_cache()

    for index, (model_dict, outputs) in enumerate(zip(model_dicts, list_probs)):
        if index == 0:  # if 'probs_total' not in locals().keys():
            ensemble_probs = probs * model_dict['model_weight']
            total_weights = model_dict['model_weight']
        else:
            ensemble_probs += probs * model_dict['model_weight']
            total_weights += model_dict['model_weight']

    ensemble_probs /= total_weights
    ensemble_y_preds = probs.argmax(axis=-1)

    return ensemble_probs, ensemble_y_preds, list_probs, list_y_preds



@torch.inference_mode()
def predict_one_image(model, img_file, image_shape=(299, 299), argmax=False, use_amp=True):
    assert os.path.exists(img_file), f'{img_file} does not exists!'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    if torch.cuda.device_count() > 0 and (not next(model.parameters()).is_cuda):
        model.to(device)

    inputs = img_to_tensor(img_file, image_shape)
    inputs = inputs.to(device)

    with autocast(enabled=use_amp):
        logits = model(inputs)
        probs = F.softmax(logits, dim=1).squeeze(dim=0)  # batch contains one sample

    if argmax:
        preds = torch.argmax(probs)
        return probs.cpu().numpy(), preds.cpu().numpy()
    else:
        return probs.cpu().numpy()




if __name__ == "__main__":  #test code
    from pathlib import Path
    from libs.dataset.my_dataset import Dataset_multiclass
    from torch.utils.data import DataLoader
    import pandas as pd
    from sklearn.metrics import confusion_matrix

    # model_name = 'inception_v3'
    # model_file1 = Path('/disk_code/code/lung_colon_cancer_classification/trained_models/LC25000_cv0/inception_v3.pth')
    # image_shape = get_input_shape(model_name)
    # model1 = create_model(model_name=model_name, num_classes=3, state_dict_file=model_file1)

    model_name = 'cspresnet50'
    model_file1 = Path('/disk_code/code/lung_colon_cancer_classification/trained_models/LC25000_cv0/cspresnet50.pth')
    from libs.neuralNetworks.utils.my_model import load_model, get_model_shape
    image_shape = get_model_shape(model_name)
    model1 = load_model(model_file=model_file1)

    csv_test = Path(__file__).resolve().parent.parent.parent.parent / 'datafiles' / f'LC25000_cv2_test.csv'
    df = pd.read_csv(csv_test)
    list_images, list_labels = df['images'].tolist(), df['labels'].tolist()

    ds_test = Dataset_multiclass(csv_file=csv_test, image_shape=image_shape)
    loader_test = DataLoader(ds_test, batch_size=64, num_workers=8, pin_memory=True)

    probs, y_preds = predict_single_model(model1, loader_test, use_amp=True)
    cm = confusion_matrix(y_pred=y_preds, y_true=list_labels)

    print(cm)
