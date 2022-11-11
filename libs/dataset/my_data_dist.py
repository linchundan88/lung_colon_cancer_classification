'''

'''
import pandas as pd



def get_class_distribution_mc(csv_file):
    df1 = pd.read_csv(csv_file)
    num_classes = len(df1['labels'].unique())
    return num_classes,  df1['labels'].tolist(), df1.groupby(['labels'])['labels'].count()