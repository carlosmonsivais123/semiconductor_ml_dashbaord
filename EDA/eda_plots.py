import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from feature_engine.selection import SmartCorrelatedSelection

class Exploratory_Data_Analysis:
    def label_count(self, df, plot_location):
        label_counts=pd.DataFrame(df.groupby('Label')['Label'].count()).\
            rename(columns={'Label': 'Label Count'}).reset_index(drop=False)
        label_counts['Label']=label_counts['Label'].astype(str)
        
        label_count_barplot=px.bar(label_counts, 
                                   x='Label', 
                                   y='Label Count', 
                                   title='Label Count',
                                   color='Label')
        label_count_barplot.update_layout(title_x=0.5)
        label_count_barplot.write_image(f"{plot_location}/label_counts.png")


    def labels_over_time(self, df, plot_location):
        df['Label']=df['Label'].astype(str)
        labels_over_time=px.scatter(df, 
                                    x='Time', 
                                    y='Label', 
                                    title='Labels Over Time',
                                    color='Label')
        labels_over_time.update_layout(title_x=0.5)
        labels_over_time.write_image(f"{plot_location}/labels_over_time.png")


    def correlation_plot(self, df, plot_location):
        df_corr=df[df.columns[~df.columns.isin(['Label', 'Time', 'day_of_week'])]].corr()

        mask=np.triu(np.ones_like(df_corr, dtype=bool))

        f, ax=plt.subplots(figsize=(20, 20))
        sns.heatmap(df_corr, mask=mask, cmap='YlGnBu')
        plt.title('Correlation Heatmap: All Data', fontsize = 20)
        plt.savefig(f"{plot_location}/correlation_heatmap_all_data.png")


    def remove_highly_correlated_features(self, df, correlation_threshold, clean_data_location):
        print(f'Before removing features with over an {correlation_threshold*100}% correlation the data had a shape of {df.shape}')

        corr_vars_list=[ele for ele in df.columns.tolist() if ele not in ['Label', 'Time', 'day_of_week']]

        df=SmartCorrelatedSelection(threshold=correlation_threshold,
                                    variables=corr_vars_list, 
                                    selection_method='missing_values').\
                                        fit_transform(df)
        
        print(f'After removing features with over an {correlation_threshold*100}% correlation the data has a shape of {df.shape}')

        df.to_csv(clean_data_location, index=False, header=False)

        print(f'The clean data has been output to: {clean_data_location}\n')
        
        return df
    

    def correlation_plot_after_removing_high_corr(self, df, plot_location):
        df_corr=df[df.columns[~df.columns.isin(['Label', 'Time', 'day_of_week'])]].corr()

        mask=np.triu(np.ones_like(df_corr, dtype=bool))

        f, ax=plt.subplots(figsize=(20, 20))
        sns.heatmap(df_corr, mask=mask, cmap='YlGnBu')
        plt.title('Correlation Heatmap: After Removing Highly Correlated Variables', fontsize = 20)
        plt.savefig(f"{plot_location}/correlation_heatmap_after_removing_high_corr_vars.png")


    def label_counts_by_dow(self, df, plot_location):
        labels_by_dow=df.groupby(['day_of_week', 'Label'])['Time'].count().reset_index(drop=False)
        labels_by_dow=labels_by_dow.rename(columns={'Time': 'Count'})
        labels_by_dow['Percentage']=(labels_by_dow['Count']/labels_by_dow['Count'].sum()) * 100

        dow_labels_barplot=px.bar(labels_by_dow, 
                                  x="day_of_week",
                                  y="Percentage", 
                                  color="Label", 
                                  title='Percentage of Label Occurence by Day of Week',
                                  barmode='group')
        dow_labels_barplot.update_layout(title_x=0.5)
        dow_labels_barplot.write_image(f"{plot_location}/label_barplot_by_dow.png")