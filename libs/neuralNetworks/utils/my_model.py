'''
pytorch timm
https://github.com/rwightman/pytorch-image-models
https://rwightman.github.io/pytorch-image-models/feature_extraction/'''
import torch
import timm



def get_model_shape(model_name):
    if model_name in ['xception', 'inception_resnet_v2', 'inception_v3']:
        image_shape = (299, 299)
    elif model_name in ['vit_base_patch16_384', 'vit_large_patch16_384', 'deit_base_patch16_384']:
        image_shape = (384, 384)
    elif '_224' in model_name:
        image_shape = (224, 224)
    elif '_256' in model_name:
        image_shape = (256, 256)
    elif '_384' in model_name:
        image_shape = (384, 384)
    elif 'efficientnet_b0' in model_name:
        image_shape = (224, 224)
    elif 'efficientnet_b1' in model_name:
        image_shape = (240, 240)
    elif 'efficientnet_b2' in model_name:
        image_shape = (260, 260)
    elif 'efficientnet_b3' in model_name:
        image_shape = (300, 300)
    elif 'efficientnet_b4' in model_name:
        image_shape = (380, 380)
    elif 'efficientnet_b5' in model_name:
        image_shape = (456, 456)
    elif 'efficientnet_b6' in model_name:
        image_shape = (528, 528)
    elif 'efficientnet_b7' in model_name:
        image_shape = (600, 600)
    elif 'efficientnetv2_b0' in model_name:
        image_shape = (224, 224)
    elif 'efficientnetv2_b1' in model_name:
        image_shape = (240, 240)
    elif 'efficientnetv2_b2' in model_name:
        image_shape = (260, 260)
    elif 'efficientnetv2_b3' in model_name:
        image_shape = (300, 300)
    elif 'efficientnetv2_s' in model_name:
        image_shape = (384, 384)
    elif 'efficientnetv2_m' in model_name:
        image_shape = (480, 480)
    elif 'efficientnetv2_l' in model_name:
        image_shape = (480, 480)
    else:
        image_shape = (224, 224)

    return image_shape


def load_model(model_file):
    print(f'load model from:{model_file}...')
    model = torch.load(model_file)
    print(f'load model:{model_file} completed!')
    return model


def create_model(model_name, num_classes=2, state_dict_file=None):
    print(f'creating model:{model_name}...')
    model = timm.create_model(model_name, num_classes=num_classes, pretrained=True) #features_only=False
    print(f'creating model:{model_name} completed.')

    if state_dict_file is not None:
        state_dict = torch.load(state_dict_file, map_location='cpu')
        model.load_state_dict(state_dict)

    return model



if __name__ == "__main__":  #test code
    model_name = 'swinv2_base_window8_256'
    model_file = '/disk_code/code/lung_colon_cancer_classification/trained_models_2022_7_31/LC25000_cv3/swinv2_base_window8_256_times0/valid_loss_0.0012_epoch10.pth'
    image_shape = get_model_shape(model_name)
    model, image_shape = create_model(model_name, num_classes=3, state_dict_file=model_file)

    print('OK')