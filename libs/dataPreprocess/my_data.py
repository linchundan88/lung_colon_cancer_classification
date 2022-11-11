'''
  write_csv
  write_csv_based_on_dir
  split_dataset
'''
import os
import random
import pandas as pd
import csv
import sklearn


def write_csv_based_on_dir(filename_csv, base_dir, dict_mapping, match_type='header',
       list_file_ext=['.BMP', '.PNG', '.JPG', '.JPEG', '.TIFF', '.TIF']):

    assert match_type in ['header', 'partial', 'end'], 'match type is error'

    if os.path.exists(filename_csv):
        os.remove(filename_csv)
    os.makedirs(os.path.dirname(filename_csv), exist_ok=True)

    if not base_dir.endswith('/'):
        base_dir = base_dir + '/'

    with open(filename_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(['images', 'labels'])

        for dir_path, subpaths, files in os.walk(base_dir, False):
            for f in files:
                img_file_source = os.path.join(dir_path, f)
                (filedir, tempfilename) = os.path.split(img_file_source)
                (filename, extension) = os.path.splitext(tempfilename)
                if extension.upper() not in list_file_ext:
                    print('file ext name:', f)
                    continue

                if not filedir.endswith('/'):
                    filedir += '/'

                for (k, v) in dict_mapping.items():
                    if match_type == 'header':
                        dir1 = os.path.join(base_dir, k)
                        if not dir1.endswith('/'):
                            dir1 += '/'

                        if dir1 in filedir:
                            print(f'writing record:{img_file_source}')
                            csv_writer.writerow([img_file_source, v])
                            break
                    elif match_type == 'partial':
                        if '/' + k + '/' in filedir:
                            print(f'writing record:{img_file_source}')
                            csv_writer.writerow([img_file_source, v])
                            break
                    elif match_type == 'end':
                        if filedir.endswith('/' + k + '/'):
                            print(f'writing record:{img_file_source}')
                            csv_writer.writerow([img_file_source, v])
                            break

    print(f'write csv file {filename_csv} completed.')



def write_csv(files, labels, filename_csv, field_columns=['images', 'labels']):
    if os.path.exists(filename_csv):
        os.remove(filename_csv)

    os.makedirs(os.path.dirname(filename_csv), exist_ok=True)

    with open(filename_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow([field_columns[0], field_columns[1]])

        for i, file in enumerate(files):
            csv_writer.writerow([file, labels[i]])

    print(f'write csv file {filename_csv} completed!')



def split_dataset(filename_csv, valid_ratio=0.1, test_ratio=None,
                  shuffle=True, random_state=None, field_columns=['images', 'labels']):

    if filename_csv.endswith('.csv'):
        df = pd.read_csv(filename_csv)
    elif filename_csv.endswith('.xls') or filename_csv.endswith('.xlsx'):
        df = pd.read_excel(filename_csv)

    if shuffle:
        df = sklearn.utils.shuffle(df, random_state=random_state)

    if test_ratio is None:
        split_num_train = int(len(df)*(1-valid_ratio))
        df_train = df[:split_num_train]
        train_files = df_train[field_columns[0]].tolist()
        train_labels = df_train[field_columns[1]].tolist()

        df_valid = df[split_num_train:]
        valid_files = df_valid[field_columns[0]].tolist()
        valid_labels = df_valid[field_columns[1]].tolist()

        return train_files, train_labels, valid_files, valid_labels
    else:
        split_num_train = int(len(df) * (1 - valid_ratio - test_ratio))
        df_train = df[:split_num_train]
        train_files = df_train[field_columns[0]].tolist()
        train_labels = df_train[field_columns[1]].tolist()

        split_num_valid = int(len(df) * (1 - test_ratio))
        df_valid = df[split_num_train:split_num_valid]
        valid_files = df_valid[field_columns[0]].tolist()
        valid_labels = df_valid[field_columns[1]].tolist()

        df_test = df[split_num_valid:]
        test_files = df_test[field_columns[0]].tolist()
        test_labels = df_test[field_columns[1]].tolist()

        return train_files, train_labels, valid_files, valid_labels, test_files, test_labels
