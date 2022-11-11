#!/bin/bash

PATH_SAVE="/disk_code/code/lung_colon_cancer_classification/trained_models_5_classes/2022_11_2"
GPU_DEVICES="0,1"
TRAIN_TIMES=4


for ((i=0;i<TRAIN_TIMES;i++))
do
#  for task_type in "LC25000_cv0" "LC25000_cv1" "LC25000_cv2"  "LC25000_cv3"  "LC25000_cv4" # 5 fold crossvalidation
  for task_type in "LC25000_5_classes_cv0" "LC25000_5_classes_cv1" "LC25000_5_classes_cv2"  "LC25000_5_classes_cv3"  "LC25000_5_classes_cv4" # 5 fold crossvalidation
  do
    #convit_small
    #beit_base_patch16_224 beit_base_patch16_224_in22k BEiT-v2
    #resnetv2_101x1_bitm big transfer
    #"inception_resnet_v2"  "vit_small_patch16_224" "resnetv2_101x1_bitm"
    #regnety_016 res2net50_26w_4s res2net50_26w_6s regnetx008  tf_efficientnetv2_b0  'tf_efficientnetv2_m_in21ft1k',
    #swin_base_patch4_window7_224  swin_base_384 swin2_base_224 swin2_base_384 swin2_large_384
    #"mobilenetv3_large_100" "mobilenetv2_100" "densenet121" "deit_base_patch16_224"
    #"resnetv2_101x1_bitm" "inception_resnet_v2" "xception"  "inception_v3" "resnetv2_101x1_bitm" "efficientnet_b3" "tf_efficientnetv2_b3" "mobilenetv3_large_100" "mobilenetv2_100" "densenet121" "res2net50_26w_6s" "regnety_032"

#    for model_name in "deit_small_patch16_224" "deit_base_patch16_224"  #"deit3_small_patch16_224"
     for model_name in "convit_small" "convit_base"  "convit_tiny"
#    for model_name in "densenet121" #"mobilenetv2_100"  "mobilenetv2_110d" "mobilenetv2_120d" "mobilenetv3_small_075" "mobilenetv3_large_100_miil_in21k"
#    for model_name in "beit_base_patch16_224_in22k"  #regular Vision Transformers, but pre-trained in a self-supervised
    #for model_name in  "deit_base_patch16_384" "deit_base_patch16_224" "swin_base_patch4_window7_224" "swin_base_patch4_window12_384"  "swinv2_base_window8_256" "swinv2_base_window16_256"  "xcit_medium_24_p8_224" "xcit_medium_24_p8_384_dist"
#    for model_name in  "resnetv2_50x1_bitm_in21k"  #"efficientnet_b2" "tf_efficientnetv2_b2"
    #for model_name in  "inception_resnet_v2" "xception" "inception_v3" "resnetv2_50x1_bitm_in21k" "resnetv2_101x1_bitm_in21k" "swin_base_patch4_window7_224" "swinv2_small_window8_256" "efficientnet_b3" "efficientnetv2_b2" "efficientnetv2_b3" "tf_efficientnetv2_m_in21ft1k"
    do
      batch_size=64
      epochs_num=40 #60 for vit, 25 for others

      if [[ $model_name == "deit_base_patch16_384" || $model_name == "vit_base_patch8_224" || $model_name == "vit_base_patch8_224_in21k"  ]] ;  #|| $model_name == "vit_small_patch16_224"
      then
        batch_size=32
      fi

      echo "training ${task_type} times:${i} model_name:${model_name} --model_type:${model_type} epochs_num:${epochs_num} batch_size:${batch_size} lr:${lr} "
      python ./my_train.py --task_type ${task_type} --model_name ${model_name}  \
            --epochs_num ${epochs_num} --use_amp  --batch_size ${batch_size}  \
            --parallel_mode DDP --gpu_devices ${GPU_DEVICES} \
            --path_save ${PATH_SAVE}/${task_type}/${model_name}_times${i}

    done

  done

done
