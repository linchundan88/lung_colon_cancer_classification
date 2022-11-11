'''
  generating csv file based on image files
  training, validation and test dataset split.
  5 folds cross validation.
'''
import pandas as pd
from libs.dataPreprocess.my_data import write_csv_based_on_dir, write_csv
from pathlib import Path
from sklearn.model_selection import KFold
import sklearn


# task_type ='LC25000'  # only focus on lung cancer
# dict_mapping = {'lung_n': 0, 'lung_scc': 1, 'lung_aca': 2}
# path_images ='/disk_data/data/LC25000/lung_image_sets'

task_type ='LC25000_5_classes'
dict_mapping = {'lung_n': 0, 'lung_scc': 1, 'lung_aca': 2, 'colon_n': 3, 'colon_aca': 4}
path_images ='/disk_data/data/LC25000_5_classes'

path_csv = Path(__file__).resolve().parent.parent / 'datafiles'
path_csv.mkdir(parents=True, exist_ok=True)

csv_all = path_csv / f'{task_type}_all.csv'
write_csv_based_on_dir(str(csv_all), path_images, dict_mapping, match_type='header')


df = pd.read_csv(csv_all)
df = sklearn.utils.shuffle(df, random_state=11111)

num_fold = 5
kf = KFold(n_splits=num_fold, shuffle=True, random_state=2222)
for index, result in enumerate(kf.split(df)):
    df_train_and_valid = df.iloc[result[0]]
    df_test = df.iloc[result[1]]

    split_num_train = int(len(df_train_and_valid) * (1 - 1 / num_fold))
    df_train = df_train_and_valid[:split_num_train]
    df_valid = df_train_and_valid[split_num_train:]

    csv_train = path_csv / f'{task_type}_cv{index}_train.csv'
    csv_valid = path_csv / f'{task_type}_cv{index}_valid.csv'
    csv_test = path_csv / f'{task_type}_cv{index}_test.csv'

    write_csv(df_train['images'].tolist(), df_train['labels'].tolist(), str(csv_train), field_columns=['images', 'labels'])
    write_csv(df_valid['images'].tolist(), df_valid['labels'].tolist(), str(csv_valid), field_columns=['images', 'labels'])
    write_csv(df_test['images'].tolist(), df_test['labels'].tolist(), str(csv_test), field_columns=['images', 'labels'])



print('OK')