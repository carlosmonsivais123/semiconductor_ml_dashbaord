from Configuration_Variables.read_config import secom_data_values_location, secom_data_labels_location, \
                                                eda_visualizatiion_graph_output, \
                                                missing_value_threshold
from Read_Data.read_data import Read_And_Merge_Data
from Clean_Data.data_cleaning import Clean_Data
from EDA.eda_plots import Exploratory_Data_Analysis

read_and_merge_data=Read_And_Merge_Data(secom_data_values_location=secom_data_values_location, 
                                        secom_data_labels=secom_data_labels_location)
clean_data=Clean_Data()
exploratory_data_analysis=Exploratory_Data_Analysis()

# Reading Merged Data
data_df=read_and_merge_data.merge_values_and_labels()

# Cleaning Data
clean_data.all_missing_values_visualizations(df=data_df, 
                                             plot_location=eda_visualizatiion_graph_output)
clean_data.missing_values_by_dow(df=data_df, 
                                 plot_location=eda_visualizatiion_graph_output)

clean_df=clean_data.remove_cols_over_missing_threhsold_val(df=data_df,
                                                           missing_value_threshold=missing_value_threshold)

clean_data.all_missing_values_after_dropped_cols_visualizations(df=clean_df, 
                                                                plot_location=eda_visualizatiion_graph_output)

# EDA
exploratory_data_analysis.label_count(df=clean_df, 
                                      plot_location=eda_visualizatiion_graph_output)