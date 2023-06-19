import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="data_cleaning",
                   page_icon='🧹')

st.write("# Data Cleaning Process 🧹")

st.sidebar.success("Data Cleaning Process 🧹")

original_data_df=pd.read_csv('/Users/carlosmonsivais/Desktop/secom/data/merged_original_data.csv')

# Chart 1
# original_data_missing_values=plt.figure(figsize=(10,6))
original_data_missing_values=plt.figure()
sns.heatmap(original_data_df.isna(),
            cmap="YlGnBu",
            cbar_kws={'label': 'Missing Data'}).set(title='Missing Values Heatmap: All Data',
                                                    xlabel='Columns', 
                                                    ylabel='Rows')
st.pyplot(original_data_missing_values)


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
st.plotly_chart(missing_dow_fig)

# container1 = st.container()
# col1, col2 = st.columns([1, 3])
# col1, col2 = st.columns(1)

# with container1:
# with col1:
#     st.pyplot(original_data_missing_values, use_container_width=True)
# with col2:
#     st.plotly_chart(missing_dow_fig, use_container_width=True)

# col1, col2 = st.columns(2)

# col1.pyplot(original_data_missing_values, use_column_width=True)

# col2.pyplot(missing_dow_fig, use_column_width=True)