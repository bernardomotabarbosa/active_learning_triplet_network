import pandas as pd
import os

path = 'results/dfs/'
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
df_rf['value_min'] = col.min(axis=1)
df_rf['value_mean'] = col.mean(axis=1)
df_rf['value_max'] = col.max(axis=1)
df_rf['value_var'] = col.var(axis=1)
df_rf = df_rf.drop(['value', 'value2', 'value3', 'value4', 'value5'], axis=1)

train_size = df_rf['train size'].unique().tolist()

# Tables
df_entropy = df_rf[df_rf['classifier name'] == f'tripletnet + RF [entropy]']
df_random = df_rf[df_rf['classifier name'] == f'tripletnet + RF [random]']
df_top_two_margin = df_rf[df_rf['classifier name'] == f'tripletnet + RF']
df_entropy = df_entropy.drop(['classifier name'], axis=1)
df_random = df_random.drop(['classifier name'], axis=1)
df_top_two_margin = df_top_two_margin.drop(['classifier name'], axis=1)

df_entropy.round(4).to_csv('results/metrics_per_step/df_entropy.csv', index=False)
df_random.round(4).to_csv('results/metrics_per_step/df_random.csv', index=False)
df_top_two_margin.round(4).to_csv('results/metrics_per_step/df_top_two_margin.csv', index=False)

# Graph
entropy = df_rf['value_mean'][df_rf['classifier name'] == f'tripletnet + RF [entropy]'].tolist()
random = df_rf['value_mean'][df_rf['classifier name'] == f'tripletnet + RF [random]'].tolist()
top_two_margin = df_rf['value_mean'][df_rf['classifier name'] == f'tripletnet + RF'].tolist()
df_rf = pd.DataFrame(columns=['entropy', 'random', 'top_two_margin'], index=train_size)
df_rf['entropy'] = entropy
df_rf['random'] = random
df_rf['top_two_margin'] = top_two_margin
lines = df_rf.plot.line()
