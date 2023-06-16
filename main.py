from Configuration_Variables.read_config import secom_data_values_location, secom_data_labels_location, \
                                                eda_visualizatiion_graph_output, \
                                                missing_value_threshold, unique_value_threshold, \
                                                correlation_drop_threshold
from Read_Data.read_data import Read_And_Merge_Data
from Clean_Data.data_cleaning import Clean_Data
from EDA.eda_plots import Exploratory_Data_Analysis

read_and_merge_data=Read_And_Merge_Data(secom_data_values_location=secom_data_values_location, 
                                        secom_data_labels=secom_data_labels_location)
clean_data=Clean_Data()
exploratory_data_analysis=Exploratory_Data_Analysis()


################## Cleaning Data ##################
# Merging Values and Labels Into a Single Dataset
data_df=read_and_merge_data.merge_values_and_labels()

# Missing Values Heatmap: All Data
clean_data.all_missing_values_visualizations(df=data_df, 
                                             plot_location=eda_visualizatiion_graph_output)

# Missing Values by Day of the Week Barplot
clean_data.missing_values_by_dow(df=data_df, 
                                 plot_location=eda_visualizatiion_graph_output)

# Removing Features with Missing Value Percetnage Over Threshold
clean_df=clean_data.remove_cols_over_missing_threhsold_val(df=data_df,
                                                           missing_value_threshold=missing_value_threshold)

# Removing Features with Unique Value Threshold
clean_df=clean_data.remove_columns_with_n_unique_vals(df=clean_df,
                                                      unique_value_threshold=unique_value_threshold)

# Missing Values Heatmap: After Removing Features from Missing Value Percentage and Unique Value Threshold
clean_data.all_missing_values_after_dropped_cols_visualizations(df=clean_df, 
                                                                plot_location=eda_visualizatiion_graph_output)


################## EDA ##################
# Label Count Barplot
exploratory_data_analysis.label_count(df=clean_df, 
                                      plot_location=eda_visualizatiion_graph_output)

# Labels Over Time Scatterplot
exploratory_data_analysis.labels_over_time(df=clean_df,
                                           plot_location=eda_visualizatiion_graph_output)

# Correlation Heatmap: All Data
exploratory_data_analysis.correlation_plot(df=clean_df,
                                           plot_location=eda_visualizatiion_graph_output)

# Removing Highly Correlated Features
clean_df_corr_drop=exploratory_data_analysis.remove_highly_correlated_features(df=clean_df, 
                                                                               correlation_threshold=correlation_drop_threshold)

# Correlation Heatmap: After Removing Highly Correlated Features
exploratory_data_analysis.correlation_plot_after_removing_high_corr(df=clean_df_corr_drop,
                                                                    plot_location=eda_visualizatiion_graph_output)

# Label Counts by Day of the Week Barplot
exploratory_data_analysis.label_counts_by_dow(df=clean_df_corr_drop, 
                                              plot_location=eda_visualizatiion_graph_output)


################## Splitting Data ##################