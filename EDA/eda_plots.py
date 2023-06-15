import pandas as pd
import plotly.express as px

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

    def labels_by_dow(self, df, plot_location):
        return None

    def correlation_plot(self, df, plot_location):
        return None