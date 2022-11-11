import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# list_models = ['swin_base_patch4_window7_224', 'swinv2_small_window8_256',
#                'inception_resnet_v2', 'xception', 'inception_v3', 'densenet121',
#                ]

list_models = ['efficientnet_b2', 'efficientnet_b3', 'tf_efficientnetv2_b3', 'inception_resnet_v2', 'xception', 'inception_v3', 'densenet121',
               'resnetv2_101x1_bitm_in21k',
               'vit_small_patch16_224', 'vit_base_patch16_224_in21k',
               'convit_base', 'convit_small', 'convit_tiny',
               'deit_base_patch16_224', 'deit_small_patch16_224',
               'swin_base_patch4_window7_224', 'swinv2_small_window8_256',
               'mobilenetv2_100', 'mobilenetv2_110d', 'mobilenetv2_120d', 'mobilenetv3_large_100_miil_in21k', 'mobilenetv3_small_075']


for model_name in list_models:
    print(f'model name:{model_name}')
    cm = pickle.load(open(Path('/disk_code/code/lung_colon_cancer_classification/results') / f'{model_name}_cm.pkl', 'rb'))

    # from sklearn.metrics import ConfusionMatrixDisplay
    # cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['lung_n', 'lung_scc', 'lung_aca'], )
    # cmd.plot(cmap="Blues")  #default viridis
    # cmd.ax_.set(xlabel='Predicted', ylabel='True')

    if model_name == 'inception_v3':
        model_name = 'Inception V3'
    if model_name == 'inception_resnet_v2':
        model_name = 'Inception-Resnet V2'
    if model_name == 'resnetv2_101x1_bitm_in21k':
        model_name = 'Resnetv2_101x1_bitm'
    if model_name == 'efficientnet_b2':
        model_name = 'EfficientNetB2'
    if model_name == 'efficientnet_b3':
        model_name = 'EfficientNetB3'
    if model_name == 'densenet121':
        model_name = 'DenseNet121'
    if model_name == 'mobilenetv2_100':
        model_name = 'MobileNetV2 100'
    if model_name == 'mobilenetv2_110d':
        model_name = 'MobileNetV2_100d'
    if model_name == 'mobilenetv2_120d':
        model_name = 'MobileNetV2 120d'
    if model_name == 'mobilenetv3_small_075':
        model_name = 'MobileNetV3 small075'
    if model_name == 'mobilenetv3_large_100_miil_in21k':
        model_name = 'MobileNetV3 large100'
    if model_name == 'vit_small_patch16_224':
        model_name = 'ViT small'
    if model_name == 'vit_base_patch16_224_in21k':
        model_name = 'ViT base'
    if model_name == 'deit_small_patch16_224':
        model_name = 'Deit small'
    if model_name == 'deit_base_patch16_224':
        model_name = 'Deit base'
    if model_name == 'convit_tiny':
        model_name = 'Convit tiny'
    if model_name == 'convit_small':
        model_name = 'Convit small'
    if model_name == 'convit_base':
        model_name = 'Convit base'
    if model_name == 'swin_base_patch4_window7_224':
        model_name = 'Swin Transformer base'
    if model_name == 'swinv2_small_window8_256':
        model_name = 'Swin Transformer V2 base'


    ax = sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues')  # annot=True to annotate cells, fmt='g' to disable scientific notation
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(f'Confusion Matrix of Model:{model_name}')
    ax.xaxis.set_ticklabels(['lung_n', 'lung_scc', 'lung_aca', 'colon_n', 'colon_aca'])
    ax.yaxis.set_ticklabels(['lung_n', 'lung_scc', 'lung_aca', 'colon_n', 'colon_aca'])
    plt.savefig(Path('/disk_code/code/lung_colon_cancer_classification/results') / f'{model_name}_cm.png')  # , bbox_inches='tight'
    plt.close()

print('OK')