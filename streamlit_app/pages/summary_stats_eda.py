import streamlit as st
import pandas as pd

import plotly.express as px

st.set_page_config(page_title="Summary Stats EDA",
                   page_icon='📊')

st.write("# Summary Stats EDA 📊")

st.sidebar.success("Summary Stats EDA 📊")

clean_data_df=pd.read_csv('/Users/carlosmonsivais/Desktop/secom/data/clean_data.csv')

label_counts=pd.DataFrame(clean_data_df.groupby('Label')['Label'].count()).\
    rename(columns={'Label': 'Label Count'}).reset_index(drop=False)
label_counts['Label']=label_counts['Label'].astype(str)

label_count_barplot=px.bar(label_counts, 
                            x='Label', 
                            y='Label Count', 
                            title='Label Count',
                            color='Label')
label_count_barplot.update_layout(title_x=0.5)


clean_data_df['Label']=clean_data_df['Label'].astype(str)
labels_over_time=px.scatter(clean_data_df, 
                            x='Time', 
                            y='Label',
                            title='Labels Over Time',
                            color='Label')
labels_over_time.update_layout(title_x=0.5)

container1 = st.container()
col1, col2 = st.columns(2)

with container1:
    with col1:
        st.plotly_chart(label_count_barplot, use_container_width=True)
    with col2:
        st.plotly_chart(labels_over_time, use_container_width=True)
