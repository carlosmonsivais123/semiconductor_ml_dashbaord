import yaml

with open('/Users/carlosmonsivais/Desktop/secom/Configuration_Variables/config.yaml', 'r') as file:
    input_vars=yaml.safe_load(file)

# Read Data Location
secom_data_values_location=input_vars['Data_Location']['secom_data_values_location']
secom_data_labels_location=input_vars['Data_Location']['secom_data_labels_location']

# EDA Visualizatiion Graph Output
eda_visualizatiion_graph_output=input_vars['EDA_Visualizatiion_Graph_Output']

# Missing Values Threshold
missing_value_threshold=input_vars['Missing_Value_Threshold']

# Unique Value Threshold
unique_value_threshold=input_vars['Unique_Value_Threshold']

# Correlation Drop Threshold
correlation_drop_threshold=input_vars['Correlation_Drop_Threshold']