import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Data Cleaning Process",
                   layout="wide")

st.write("# Data Cleaning Process")

st.sidebar.success("Data Cleaning Process")

original_data_df=pd.read_csv('/Users/carlosmonsivais/Desktop/secom/data/merged_original_data.csv')
data_before_imputation_df=pd.read_csv('/Users/carlosmonsivais/Desktop/secom/data/before_data_imputation.csv')
clean_data_df=pd.read_csv('/Users/carlosmonsivais/Desktop/secom/data/clean_data.csv')

st.write("#### Data Before Pre-Processing")
# Chart 1
df_na_original=original_data_df.isna()
df_na_original=df_na_original.replace({False: 0, True: 1})
original_data_missing_values=go.Figure(go.Heatmap(z=df_na_original.values,
                                                  x=df_na_original.columns,
                                                  y=df_na_original.columns,
                                                  zmin=0,
                                                  zmax=1))
original_data_missing_values.update_layout(title='Original Data Missing Values Heatmap',
                                           title_x=0.30)

# Chart 2
missing_values_by_dow=original_data_df.set_index('day_of_week').\
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
missing_dow_fig.update_layout(title_x=0.25,
                              margin=dict(l=0, r=0, b=0))

container1 = st.container()
col1, col2 = st.columns(2)

with container1:
    with col1:
        st.plotly_chart(original_data_missing_values, use_container_width=True)
    with col2:
        st.plotly_chart(missing_dow_fig, use_container_width=True)


st.write("#### Data After Pre-Processing, but before Imputation")
# Chart 1
df_na_before_imputation=data_before_imputation_df.isna()
df_na_before_imputation=df_na_before_imputation.replace({False: 0, True: 1})
before_imputation_data_missing_values=go.Figure(go.Heatmap(z=df_na_before_imputation.values,
                                                           x=df_na_before_imputation.columns,
                                                           y=df_na_before_imputation.columns,
                                                           zmin=0,
                                                           zmax=1))
before_imputation_data_missing_values.update_layout(title='Before Imputation Data Missing Values Heatmap',
                                                    title_x=0.30)

# Chart 2
before_imputation_missing_values_by_dow=data_before_imputation_df.set_index('day_of_week').\
    isna().groupby(level=0).sum().sum(axis=1).\
        reset_index(drop=False).rename(columns={'day_of_week': 'Day of Week',
                                                0: 'Number of Missing Rows'})

before_imputation_missing_values_by_dow['Missing Rows Percentage']=(before_imputation_missing_values_by_dow['Number of Missing Rows']\
                                                /before_imputation_missing_values_by_dow['Number of Missing Rows'].sum()) * 100
custom_dow_sort={'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4,
                 'Saturday': 5, 'Sunday': 6} 
before_imputation_missing_values_by_dow=before_imputation_missing_values_by_dow.sort_values(by=['Day of Week'], 
                                                                                            key=lambda x: x.map(custom_dow_sort))
before_imputation_missing_values_by_dow=before_imputation_missing_values_by_dow.reset_index(drop=True)

before_imputation_missing_dow_fig=px.bar(before_imputation_missing_values_by_dow, 
                                         x="Day of Week", 
                                         y="Missing Rows Percentage", 
                                         color="Day of Week", 
                                         title="Before Imputation Percentage of Missing Values by Day of the Week")
before_imputation_missing_dow_fig.update_layout(title_x=0.25,
                                                margin=dict(l=0, r=0, b=0))

container2 = st.container()
col3, col4 = st.columns(2)

with container2:
    with col3:
        st.plotly_chart(before_imputation_data_missing_values, use_container_width=True)
    with col4:
        st.plotly_chart(before_imputation_missing_dow_fig, use_container_width=True)



st.write("#### Data After Imputation")
# Chart 1
df_after_imputation=clean_data_df.isna()
df_after_imputation=df_after_imputation.replace({False: 0, True: 1})
df_after_imputation_data_missing_values=go.Figure(go.Heatmap(z=df_after_imputation.values,
                                                           x=df_after_imputation.columns,
                                                           y=df_after_imputation.columns,
                                                           zmin=0,
                                                           zmax=1))
df_after_imputation_data_missing_values.update_layout(title='After Imputation Data Missing Values Heatmap',
                                                      title_x=0.30)


# Chart 2
after_imputation_missing_values_by_dow=clean_data_df.set_index('day_of_week').\
    isna().groupby(level=0).sum().sum(axis=1).\
        reset_index(drop=False).rename(columns={'day_of_week': 'Day of Week',
                                                0: 'Number of Missing Rows'})

after_imputation_missing_values_by_dow['Missing Rows Percentage']=(after_imputation_missing_values_by_dow['Number of Missing Rows']\
                                                /after_imputation_missing_values_by_dow['Number of Missing Rows'].sum()) * 100
custom_dow_sort={'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4,
                 'Saturday': 5, 'Sunday': 6} 
after_imputation_missing_values_by_dow=after_imputation_missing_values_by_dow.sort_values(by=['Day of Week'], 
                                                                                            key=lambda x: x.map(custom_dow_sort))
after_imputation_missing_values_by_dow=after_imputation_missing_values_by_dow.reset_index(drop=True)

after_imputation_missing_dow_fig=px.bar(after_imputation_missing_values_by_dow, 
                                         x="Day of Week", 
                                         y="Missing Rows Percentage", 
                                         color="Day of Week", 
                                         title="Before Imputation Percentage of Missing Values by Day of the Week")
after_imputation_missing_dow_fig.update_layout(title_x=0.25,
                                                margin=dict(l=0, r=0, b=0))


container3 = st.container()
col5, col6 = st.columns(2)

with container2:
    with col5:
        st.plotly_chart(df_after_imputation_data_missing_values, use_container_width=True)
    with col6:
        st.plotly_chart(after_imputation_missing_dow_fig, use_container_width=True)