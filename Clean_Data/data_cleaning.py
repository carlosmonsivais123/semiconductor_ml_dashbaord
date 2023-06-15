import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Clean_Data:
    def missing_values_visualizations(self, df):
        plt.figure(figsize=(10,6))
        sns.heatmap(df.isna(),
                    cmap="YlGnBu",
                    cbar_kws={'label': 'Missing Data'})
        # plt.savefig("visualizing_missing_data_with_heatmap_Seaborn_Python.png", dpi=100)
        plt.show()

