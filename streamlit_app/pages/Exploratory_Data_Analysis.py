import streamlit as st
import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Exploratory Data Analysis",
                   layout="wide")

st.write("# Exploratory Data Analysis")

st.sidebar.success("Exploratory Data Analysis")


original_data_df=pd.read_csv('../data/merged_original_data.csv')
clean_data_df=pd.read_csv('../data/clean_data.csv')

st.write("#### Label Imbalance")
st.markdown('''We are dealing with an interesting classification problem where we can see that there is an imbalance of labels. The labels have the following meanings:
- 0: Pass (During semi-conductor in house testing) 
- 1: Fail (During semi-conductor in house testing)

With this in mind there are a lot more 0 (Pass) values over time which accounts for about 93% of cases in our data and there are approximately 7% 1 (Fail) Labels.

There does not seem to be any label imbalance issues occuring throughout the week as the levels are relatively within stable limits, there are no obvious spikes, given the scale of the data.
''')
label_counts=pd.DataFrame(clean_data_df.groupby('Label')['Label'].count()).\
    rename(columns={'Label': 'Label Count'}).reset_index(drop=False)
label_counts['Label']=label_counts['Label'].astype(str)

label_count_barplot=px.bar(label_counts, 
                            x='Label', 
                            y='Label Count', 
                            title='Label Count',
                            color='Label')
label_count_barplot.update_layout(title_x=0.40)


clean_data_df['Label']=clean_data_df['Label'].astype(str)
labels_over_time=px.scatter(clean_data_df, 
                            x='Time', 
                            y='Label',
                            title='Labels Over Time',
                            color='Label')
labels_over_time.update_layout(title_x=0.40)

container1=st.container()
col1, col2=st.columns(2)
with container1:
    with col1:
        st.plotly_chart(label_count_barplot, use_container_width=True)
    with col2:
        st.plotly_chart(labels_over_time, use_container_width=True)




labels_by_dow=clean_data_df.groupby(['day_of_week', 'Label'])['Time'].count().reset_index(drop=False)
labels_by_dow=labels_by_dow.rename(columns={'Time': 'Count'})
labels_by_dow['Percentage']=(labels_by_dow['Count']/labels_by_dow['Count'].sum()) * 100

dow_labels_barplot=px.bar(labels_by_dow, 
                            x="day_of_week",
                            y="Percentage", 
                            color="Label", 
                            title='Percentage of Label Occurence by Day of Week',
                            barmode='group')
dow_labels_barplot.update_layout(title_x=0.30)
dow_labels_barplot.update_xaxes(categoryorder='array', categoryarray= ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
st.plotly_chart(dow_labels_barplot, use_container_width=True)



st.write("#### Data Correlation")
st.markdown('''Keeping in mind correlation (in this case Spearman Correlation), is a linear measurement between features, it was interesting to see how highly correlated some features were.
This could be due to maybe there being some duplicate column values, or dependent features. As a result, to avoid multicollinearity and therefore have narrower confidence intervals for our model coefficients,
and to also not have repeated features, I removed highly correlated variables.
''')
df_corr_original=original_data_df[original_data_df.columns[~original_data_df.columns.isin(['Label', 'Time', 'day_of_week'])]].corr()
mask_original_data=np.triu(np.ones_like(df_corr_original, dtype=bool))
original_data_corr_map=go.Figure(go.Heatmap(z=df_corr_original.mask(mask_original_data),
                                            x=df_corr_original.columns,
                                            y=df_corr_original.columns,
                                            zmin=-1,
                                            zmax=1))
original_data_corr_map.update_layout(title='Original Data Correlation Heatmap',
                                     title_x=0.30)

df_corr_clean=clean_data_df[clean_data_df.columns[~clean_data_df.columns.isin(['Label', 'Time', 'day_of_week'])]].corr()
mask_clean_data=np.triu(np.ones_like(df_corr_clean, dtype=bool))
clean_data_corr_map=go.Figure(go.Heatmap(z=df_corr_clean.mask(mask_clean_data),
                                         x=df_corr_clean.columns,
                                         y=df_corr_clean.columns,
                                         zmin=-1,
                                         zmax=1))
clean_data_corr_map.update_layout(title='Clean Data Correlation Heatmap', 
                                  title_x=0.30)

container2=st.container()
col3, col4=st.columns(2)
with container2:
    with col3:
        st.plotly_chart(original_data_corr_map, use_container_width=True)
    with col4:
        st.plotly_chart(clean_data_corr_map, use_container_width=True)
