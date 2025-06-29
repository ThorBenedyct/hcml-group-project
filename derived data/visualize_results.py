import matplotlib
import pandas as pd
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

# Load data
df = pd.read_csv('data.csv', sep=';')
df['m_weight'] = df['weight'].str.extract(r"'m':\s*([0-9.]+)").astype(float)
# Group by relevant parameters excluding 'gamma'
grouped = df.groupby(['Model', 'gamma', 'kernel', 'normalization'])

# Plot each group
plt.figure(figsize=(8, 5))

for group_keys, group_df in grouped:

    # Sort by gamma for consistent plotting
    group_df_sorted = group_df.sort_values(by='m_weight')

    # Plot each accuracy metric
    plt.plot(group_df_sorted['m_weight'], group_df_sorted['Acc'], marker='o', label=f"{group_keys[2]} - {group_keys[3]}")
    #plt.plot(group_df_sorted['m_weight'], group_df_sorted['M_Acc'], marker='s', label='M_Acc', color='blue')
    #plt.plot(group_df_sorted['m_weight'], group_df_sorted['F_Acc'], marker='^', label='F_Acc', color='red')

    # Add labels and legend
title = f"Model: {group_keys[0]}, Gamma: {group_keys[1]}"
plt.title(title)
plt.xlabel('class_weight[m]')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.show()
filename = f"Model {group_keys[0]} {group_keys[1]} {group_keys[2]} {group_keys[3]}"
plt.savefig(f'{filename}.png')
