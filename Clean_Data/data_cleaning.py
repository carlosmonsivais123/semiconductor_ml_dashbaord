import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

class Clean_Data:
    def all_missing_values_visualizations(self, df, plot_location):
        plt.figure(figsize=(10,6))
        sns.heatmap(df.isna(),
                    cmap="YlGnBu",
                    cbar_kws={'label': 'Missing Data'}).set(title='Missing Values Heatmap',
                                                            xlabel='Columns', 
                                                            ylabel='Rows')
        plt.savefig(f"{plot_location}/all_missing_values.png", dpi=100)


    def missing_values_by_dow(self, df, plot_location):
        missing_values_by_dow=df.set_index('day_of_week').\
            isna().groupby(level=0).sum().sum(axis=1).\
                reset_index(drop=False).rename(columns={'day_of_week': 'Day of Week',
                                                        0: 'Number of Missing Rows'})

        missing_values_by_dow['Missing Rows Percentage']=(missing_values_by_dow['Number of Missing Rows']\
                                                        /missing_values_by_dow['Number of Missing Rows'].sum()) * 100
        custom_dow_sort={'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4,
                         'Saturday': 5, 'Sunday': 6} 
        missing_values_by_dow=missing_values_by_dow.sort_values(by=['Day of Week'], 
                                                                key=lambda x: x.map(custom_dow_sort))
        missing_values_by_dow=missing_values_by_dow.reset_index(drop=True)

        missing_dow_fig=px.bar(missing_values_by_dow, 
                               x="Day of Week", 
                               y="Missing Rows Percentage", 
                               color="Day of Week", 
                               title="Percentage of Missing Values by Day of the Week")
        missing_dow_fig.update_layout(title_x=0.5)
        missing_dow_fig.write_image(f"{plot_location}/missing_values_by_dow_percentage.png")

    
    def remove_cols_over_missing_threhsold_val(self, df, missing_value_threshold):
        data_df_missing_percent=df.isnull().sum()*100/len(df)
        data_df_missing_percent=data_df_missing_percent.reset_index(drop=False).rename(columns={'index': 'column_name', 
                                                                                                0: 'missing_value_percentage'})
        data_df_missing_percent=data_df_missing_percent.sort_values(by='missing_value_percentage',
                                                                    ascending=False).reset_index(drop=True)
        
        data_df_missing_percent=data_df_missing_percent[data_df_missing_percent['missing_value_percentage'] > missing_value_threshold]
        remove_columns=data_df_missing_percent['column_name'].values

        print(f'Originally the shape of the data is {df.shape}')
        df=df.drop(columns=remove_columns)
        print(f'Ater removing columns missing {missing_value_threshold * 100}% of data the shape of the data is {df.shape}')

        return df
    

    def all_missing_values_after_dropped_cols_visualizations(self, df, plot_location):
        plt.figure(figsize=(10,6))
        sns.heatmap(df.isna(),
                    cmap="YlGnBu",
                    cbar_kws={'label': 'Missing Data'}).set(title='Missing Values After Dropped Columns Heatmap',
                                                            xlabel='Columns', 
                                                            ylabel='Rows')
        plt.savefig(f"{plot_location}/missing_values_after_dropped_cols.png", dpi=100)