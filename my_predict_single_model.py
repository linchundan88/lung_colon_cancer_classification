
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_devices', default='0')  # 0,1
parser.add_argument('--task_type', default='LC25000_5_classes')
#recommend num_workers = the number of gpus * 4, when debugging it should be set to 0.
parser.add_argument('--use_amp', action='store_true', default=True)  # AUTOMATIC MIXED PRECISION
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--path_outputs', default='/disk_code/code/lung_colon_cancer_classification/results')
args = parser.parse_args()
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices   # setting  GPUs, must before import torch
import numpy as np
import pandas as pd
import torch
from libs.dataset.my_dataset import Dataset_multiclass
from libs.neuralNetworks.utils.my_model import load_model, get_model_shape, create_model
from torch.utils.data import DataLoader
from libs.neuralNetworks.utils.my_predict import predict_multi_models
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, cohen_kappa_score
import shutil
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':
    NUM_CLASSES = 5
    NUM_FOLD = 5

    list_models_sync_batchnorm = ['efficientnet_b2', 'efficientnet_b3', 'tf_efficientnetv2_b3', 'densenet121', 'convit_base', 'convit_small', 'convit_tiny',
                              'deit_base_patch16_224', 'deit_small_patch16_224', 'mobilenetv2_100', 'mobilenetv2_110d', 'mobilenetv2_120d',
                              'mobilenetv3_large_100_miil_in21k', 'mobilenetv3_small_075']


    list_models = ['swin_base_patch4_window7_224', 'swinv2_small_window8_256',
                   'vit_small_patch16_224', 'vit_small_patch16_224_in21k', 'vit_base_patch16_224_in21k',
                       'inception_resnet_v2', 'xception', 'inception_v3', 'densenet121',
                       'resnetv2_50x1_bitm_in21k', 'resnetv2_101x1_bitm_in21k']

    list_models = ['vit_small_patch16_224', 'vit_small_patch16_224_in21k', 'vit_base_patch16_224_in21k']
    # list_models = ['efficientnet_b2', 'efficientnet_b3', 'tf_efficientnetv2_b3', 'densenet121', 'convit_base', 'convit_small', 'convit_tiny',
    #                           'deit_base_patch16_224', 'deit_small_patch16_224', 'mobilenetv2_100', 'mobilenetv2_110d', 'mobilenetv2_120d',
    #                           'mobilenetv3_large_100_miil_in21k', 'mobilenetv3_small_075']


    for model_name in list_models:

        list_all_labels, list_all_preds = [], []
        for index in range(NUM_FOLD):
            print(f'predicting model:{model_name} for fold:{index}')
            csv_test = Path(__file__).resolve().parent / 'datafiles' / f'{args.task_type}_cv{index}_test.csv'
            df = pd.read_csv(csv_test)
            list_images, list_labels = df['images'].tolist(), df['labels'].tolist()

            model_dicts = []
            path_models = Path(__file__).resolve().parent / 'trained_models_5_classes' / (args.task_type + f'_cv{index}') #trained_models_2022_8_8
            model_file1 = path_models / f'{model_name}.pth'
            image_shape = get_model_shape(model_name)
            # model1 = load_model(model_file=model_file1)
            model1 = create_model(model_name=model_name, num_classes=NUM_CLASSES, state_dict_file=model_file1)
            # Distributed Data Parallel Training using convert_sync_batchnorm
            if model_name in list_models_sync_batchnorm:
                model1 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model1)

            ds_test = Dataset_multiclass(csv_file=csv_test, image_shape=image_shape)
            loader_test = DataLoader(ds_test, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
            model_dict1 = dict(model=model1, use_amp=args.use_amp, model_weight=1, data_loader=loader_test)
            model_dicts.append(model_dict1)

            _, ensemble_preds, _, _ = predict_multi_models(model_dicts)

            for (img_file, label_gt, label_pred) in zip(list_images, list_labels, ensemble_preds):
                if label_gt != label_pred:
                    # print(img_file)
                    path_output = Path(args.path_outputs) / model_name/ f'{label_gt}_{label_pred}'
                    path_output.mkdir(parents=True, exist_ok=True)
                    file_dest = path_output / img_file.split('/')[-1]
                    shutil.copy(Path(img_file), file_dest)

            list_all_labels.append(list_labels)
            list_all_preds.append(ensemble_preds)

        all_labels = np.vstack(list_all_labels).flatten()
        all_preds = np.vstack(list_all_preds).flatten()

        pickle.dump((all_labels, all_preds), open(Path(args.path_outputs) / f'{model_name}_cls_results.pkl', 'wb'))

        accuracy = accuracy_score(all_labels, all_preds)
        print(f'accuracy:{accuracy}')

        cm = confusion_matrix(all_labels, all_preds)
        print(cm)
        pickle.dump(cm, open(Path(args.path_outputs) / f'{model_name}_cm.pkl', 'wb'))

        cm = pickle.load(open(Path(args.path_outputs) / f'{model_name}_cm.pkl', 'rb'))


        # from sklearn.metrics import ConfusionMatrixDisplay
        # cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['lung_n', 'lung_scc', 'lung_aca'], )
        # cmd.plot(cmap="Blues")  #default viridis
        # cmd.ax_.set(xlabel='Predicted', ylabel='True')

        ax = sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues')  #annot=True to annotate cells, fmt='g' to disable scientific notation
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        ax.xaxis.set_ticklabels(['lung_n', 'lung_scc', 'lung_aca', 'colon_n', 'colon_aca'])
        ax.yaxis.set_ticklabels(['lung_n', 'lung_scc', 'lung_aca', 'colon_n', 'colon_aca'])
        plt.savefig(Path(args.path_outputs) / f'{model_name}_cm.png') #, bbox_inches='tight'
        # plt.show()
        plt.close()

        kappa = cohen_kappa_score(all_labels, all_preds)
        print(f'model:{model_name} Kappa: {kappa:.4f}')

        print(classification_report(all_labels, all_preds, digits=4))


    print('OK.')