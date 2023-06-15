from Configuration_Variables.read_config import secom_data_values_location, secom_data_labels_location
from Read_Data.read_data import Read_And_Merge_Data

read_and_merge_data=Read_And_Merge_Data(secom_data_values_location=secom_data_values_location, 
                                        secom_data_labels=secom_data_labels_location)

# Reading Merged Data
data_df=read_and_merge_data.merge_values_and_labels()

# Cleaning Data
