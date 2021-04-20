import pandas as pd
import os

path = 'results_1/dfs/'
files = os.listdir(path)

f = files[0]

initial_df = pd.read_csv(path + f)
df_f1 = initial_df[initial_df['metric name'] == 'f1_macro']
df_f1 = df_f1.drop(['metric name', 'step'], axis=1)
df_rf = df_f1.copy()

for id, f in enumerate(files[1:]):
    initial_df = pd.read_csv(path + f)
    df_f1 = initial_df[initial_df['metric name'] == 'f1_macro']
    df_f1 = df_f1.drop(['metric name', 'step'], axis=1)
    df_rf[f'value{id + 2}'] = df_f1['value']
col = df_rf.loc[:, 'value':'value5']
df_rf['value_mean'] = col.mean(axis=1)

train_size = df_rf['train size'].unique().tolist()
entropy = df_rf['value_mean'][df_rf['classifier name'] == f'tripletnet + RF [entropy]'].tolist()
random = df_rf['value_mean'][df_rf['classifier name'] == f'tripletnet + RF [random]'].tolist()
top_two_margin = df_rf['value_mean'][df_rf['classifier name'] == f'tripletnet + RF'].tolist()
df_rf = pd.DataFrame(columns=['entropy', 'random', 'top_two_margin'], index=train_size)
df_rf['entropy'] = entropy
df_rf['random'] = random
df_rf['top_two_margin'] = top_two_margin
lines = df_rf.plot.line()