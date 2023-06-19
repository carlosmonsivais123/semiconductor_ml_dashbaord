import streamlit as st
import os
import pandas as pd
import mlflow

st.set_page_config(page_title="Prediction Models",
                   layout="wide")
st.write("# Prediction Models")

st.sidebar.success("Prediction Models")

test_df=pd.read_csv('/Users/carlosmonsivais/Desktop/secom/data/testing_data.csv')

folders=os.listdir('/Users/carlosmonsivais/Desktop/secom/mlruns/0')
folders.pop()
folders.remove('.DS_Store')

metric_list=['accuracy', 'f1', 'precision', 'recall']

accuracy=[]
f1=[]
precision=[]
recall=[]

for folder in folders:
    for metric in metric_list:
        model_metrics_file=f'/Users/carlosmonsivais/Desktop/secom/mlruns/0/{folder}/metrics/{metric}'

        model_metric=pd.read_csv(f'{model_metrics_file}', sep=" ", header=None)
        
        if metric=='accuracy':
            accuracy.append(model_metric[1].values[0])

        if metric=='f1':
            f1.append(model_metric[1].values[0])

        if metric=='precision':
            precision.append(model_metric[1].values[0])

        if metric=='recall':
            recall.append(model_metric[1].values[0])

model_list=['K-Nearest Neighbors', 'Naive Bayes', 'Random Forest', 'XGBoost', 'SVM', 'Logistic Regression']
model_performance=pd.DataFrame({'Model': model_list, 'Accuracy': accuracy, 'F1': f1, 
                                'Precision': precision, 'Recall': recall})
model_performance=model_performance.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)

hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """

st.markdown(hide_table_row_index, unsafe_allow_html=True)
st.table(model_performance)

st.write("#### Based on Accuracy, F1, Precision and Recall, the best performing model is XGBoost")

model_link_dictionary={'Naive Bayes': '4d962818ecb945dfa29a7804c7e7a3e7', 'Random Forest': '8a5010e3afec4a8bbdbc6a3cfbe42f5a',
                       'XGBoost': '03274822e87f43e2aad5c7e637c81287', 'K-Nearest Neighbors': '5355331ef1dc4929bdfb73da471dad43',
                       'Logistic Regression': 'a458c1c91238456696b510b5df2d89b4', 'SVM': 'ce1993f493e54d87b761c48371a23333'}

model_name_dictionary={'Naive Bayes': 'naive_bayes_smote', 'Random Forest': 'random_forest_smote',
                       'XGBoost': 'xgboost_smote', 'K-Nearest Neighbors': 'k_nearest_neighbor_smote',
                       'Logistic Regression': 'logistic_regression_smote', 'SVM': 'support_vector_machine_smote'}


print(list(model_link_dictionary.keys()))
print(list(model_link_dictionary.values()))


option=st.selectbox('Please Select the Model For Predictions and Results',
                    ('XGBoost', 'Random Forest', 'SVM', 'Logistic Regression', 'K-Nearest Neighbors', 'Naive Bayes'))

st.write('You selected:', option, 'Model')

selected_model=mlflow.pyfunc.load_model(f'/Users/carlosmonsivais/Desktop/secom/mlruns/0/{model_link_dictionary[option]}/artifacts/{model_name_dictionary[option]}')
predictions=selected_model.predict(test_df)

print(predictions)