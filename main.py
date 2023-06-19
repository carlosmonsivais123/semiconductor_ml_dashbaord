from Configuration_Variables.read_config import secom_data_values_location, secom_data_labels_location, \
                                                merged_data_location, \
                                                before_data_imputation_location, \
                                                eda_visualizatiion_graph_output, \
                                                missing_value_threshold, unique_value_threshold, \
                                                correlation_drop_threshold, \
                                                clean_data_location, \
                                                random_state_value, \
                                                training_data_output_location, testing_data_output_location
from Read_Data.read_data import Read_And_Merge_Data
from Clean_Data.data_cleaning import Clean_Data
from EDA.eda_plots import Exploratory_Data_Analysis
from Data_Split.train_test_split_proportionally import Create_Train_Test_Split
from Model_Creation.create_classification_models import Create_Classification_Models

read_and_merge_data=Read_And_Merge_Data(secom_data_values_location=secom_data_values_location, 
                                        secom_data_labels=secom_data_labels_location)
clean_data=Clean_Data()
exploratory_data_analysis=Exploratory_Data_Analysis()
create_train_test_split=Create_Train_Test_Split()
create_classification_models=Create_Classification_Models(random_state=random_state_value)


################## Cleaning Data ##################
# Merging Values and Labels Into a Single Dataset
data_df=read_and_merge_data.merge_values_and_labels(merged_data_location=merged_data_location)

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
                                                      unique_value_threshold=unique_value_threshold,
                                                      before_data_imputation_location=before_data_imputation_location)

# Imputation Pipeline Using the Median
clean_df=clean_data.imputing_with_median(df=clean_df)

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
                                                                               correlation_threshold=correlation_drop_threshold,
                                                                               clean_data_location=clean_data_location)

# Correlation Heatmap: After Removing Highly Correlated Features
exploratory_data_analysis.correlation_plot_after_removing_high_corr(df=clean_df_corr_drop,
                                                                    plot_location=eda_visualizatiion_graph_output)

# Label Counts by Day of the Week Barplot
exploratory_data_analysis.label_counts_by_dow(df=clean_df_corr_drop, 
                                              plot_location=eda_visualizatiion_graph_output)



################## Splitting Data ##################
# Splitting into training and testing dataframes
training_df, testing_df=create_train_test_split.create_and_save_train_test_split(df=clean_df_corr_drop,
                                                                         random_state_value=random_state_value,
                                                                         training_data_output_location=training_data_output_location, 
                                                                         testing_data_output_location=testing_data_output_location)

create_train_test_split.evaluate_data_balance_for_train_test_split(train_df=training_df, 
                                                                   test_df=testing_df,
                                                                   plot_location=eda_visualizatiion_graph_output)



################## MLflow Model Creation and Storage ##################
# # Creating splits for X and y data for training and testing
# X_train, y_train, X_test, y_test=create_classification_models.create_X_and_y_data(train_df=training_df, 
#                                                                                   test_df=testing_df)

# # Modeling process using GridsearchCV and storing models in MLflow
# create_classification_models.gridsearch_cv_best_model_mlflow(train_df=X_train,
#                                                              test_df=y_train,
#                                                              X_test=X_test,
#                                                              y_test=y_test)