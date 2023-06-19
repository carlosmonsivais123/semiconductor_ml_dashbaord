import streamlit as st
import pandas as pd

st.set_page_config(page_title="Summary Stats EDA",
                   page_icon='📊')

st.write("# Summary Stats EDA 📊")

st.sidebar.success("Summary Stats EDA 📊")

original_data_df=pd.read_csv('/Users/carlosmonsivais/Desktop/secom/data/merged_original_data.csv')