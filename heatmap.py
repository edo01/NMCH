import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


file_path = '/Users/xumengqian/Desktop/data/result_10000.csv'#add the path
data = pd.read_csv(file_path)
data.columns = data.columns.str.strip()

# Exclude the 'method' column from numeric conversion
columns_to_convert = ['k', 'theta', 'sigma', 'bias']
for col in columns_to_convert:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Drop rows with NaN in relevant columns
data = data.dropna(subset=columns_to_convert)

# unique methods: FE & EM
methods = data['method'].unique()


for method in methods:
    method_data = data[data['method'] == method]
    unique_sigma = method_data['sigma'].unique()  
    num = len(unique_sigma)
    
    group_size = num // 3 + (1 if num % 3 else 0)  
    sigma_groups = [unique_sigma[i:i + group_size] for i in range(0, num, group_size)]  # 分成三组
    
    for i, sigma_group in enumerate(sigma_groups):
        print(f"Displaying group {i + 1} for method {method}")
     
        fig, axes = plt.subplots(1, len(sigma_group), figsize=(5 * len(sigma_group), 8), constrained_layout=True)
        if len(sigma_group) == 1:  
            axes = [axes]
        
        for ax, sigma_value in zip(axes, sigma_group):
            filtered_data = method_data[method_data['sigma'] == sigma_value]
            
            heatmap_data = filtered_data.pivot_table(
                index='k', columns='theta', values='bias', aggfunc='mean'
            )
            
            sns.heatmap(heatmap_data, annot=False, cmap='viridis', cbar_kws={'label': 'Bias'}, ax=ax)
            ax.set_title(f"Sigma = {sigma_value}")
            ax.set_xlabel('Theta')
            ax.set_ylabel('K')
        
        
        fig.suptitle(f"Heatmaps of Bias (Method: {method}, Group {i + 1})", fontsize=16)
        plt.show()

