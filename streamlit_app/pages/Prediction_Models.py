import streamlit as st
import os
import pandas as pd
import mlflow
from sklearn.metrics import confusion_matrix 
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Prediction Models",
                   layout="wide")
st.write("# Prediction Models")

st.sidebar.success("Prediction Models")

train_df=pd.read_csv('data/training_data.csv')
test_df=pd.read_csv('data/testing_data.csv')

folders=['5355331ef1dc4929bdfb73da471dad43', '4d962818ecb945dfa29a7804c7e7a3e7', '8a5010e3afec4a8bbdbc6a3cfbe42f5a',
         '03274822e87f43e2aad5c7e637c81287', 'ce1993f493e54d87b761c48371a23333', 'a458c1c91238456696b510b5df2d89b4']

st.write("#### Overall Model Performance")
st.markdown('''Below are the models that I tried to implement to create a solution, where they were measure on multiple metrics including:
1. Accuracy: The overall number of correct predictions.
2. F1: The average of Precision and Recall to give a different measurement of accuracy. This measures the model's ability to capture positive and negative cases.
3. Precision: The rate at which positive cases are being predicted.
4. Recall: The proportion of actual positive cases that were identified.
''')
metric_list=['accuracy', 'f1', 'precision', 'recall']

accuracy=[]
f1=[]
precision=[]
recall=[]

for folder in folders:
    for metric in metric_list:
        model_metrics_file=f'mlruns/0/{folder}/metrics/{metric}'

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


option=st.selectbox('Please Select the Model For Predictions and Results',
                    ('XGBoost', 'Random Forest', 'SVM', 'Logistic Regression', 'K-Nearest Neighbors', 'Naive Bayes'))

st.write('You selected:', option, 'Model')

selected_model=mlflow.pyfunc.load_model(f'mlruns/0/{model_link_dictionary[option]}/artifacts/{model_name_dictionary[option]}')
predictions=selected_model.predict(test_df)

predictions_df=pd.DataFrame({'Time': test_df['Time'], 'Actual': test_df['Label'], 'Prediction': predictions})
predictions_df['Time']=pd.to_datetime(predictions_df['Time'])
predictions_df=predictions_df.sort_values(by='Time', 
                                          ascending=True).\
                                            reset_index(drop=True)
# Confusion Matrix
sklearn_confusion_matrix_output=confusion_matrix(predictions_df['Actual'].values, predictions_df['Prediction'])
 
confusion_matrix_figure=ff.create_annotated_heatmap(z=sklearn_confusion_matrix_output, x=['0', '1'], y=['0', '1'], colorscale='Greens')
confusion_matrix_figure.update_layout(title_text='Confusion Matrix', 
                                      title_x=0.30)


# Actual vs Predictions Scatter Plot
predictions_scatter=go.Figure()
predictions_scatter.add_trace(go.Scatter(x=predictions_df['Time'], 
                                         y=predictions_df['Actual'],
                                         mode='markers',
                                         marker=dict(color='Green'),
                                         opacity=0.7,
                                         name='Actual'))
predictions_scatter.add_trace(go.Scatter(x=predictions_df['Time'], 
                                         y=predictions_df['Prediction'],
                                         mode='markers',
                                         marker=dict(color='Blue'),
                                         opacity=0.7,
                                         name='Prediction'))
predictions_scatter.update_layout(title_text='Actual vs Predictions Scatter Plot',
                                  title_x=0.35)

container1 = st.container()
col1, col2 = st.columns([3, 1])

with container1:
    with col1:
        st.plotly_chart(predictions_scatter, use_container_width=True)
    with col2:
        st.plotly_chart(confusion_matrix_figure, use_container_width=True)
        

# Feature Importance Graph  
st.write("#### XGBoost Model Feature Importance")
st.markdown('''Since I chose the XGBoost model as the best mode, I wanted to look at the important features in the model so that for the next iteration, 
we may only want to model on these features to reduce noise in other features.
''')
sklearn_pipeline_xgboost_model=mlflow.sklearn.load_model(f'mlruns/0/03274822e87f43e2aad5c7e637c81287/artifacts/xgboost_smote')
predictions=sklearn_pipeline_xgboost_model.predict(test_df)
xgboost_model=sklearn_pipeline_xgboost_model.named_steps['xgboost_cv_step']
feature_importance_values=pd.DataFrame(xgboost_model.best_estimator_.get_booster().get_score(importance_type='gain'), index=[0]).transpose().reset_index(drop=False)
feature_importance_values.columns=['Feature', 'Information Gain']
feature_importance_values=feature_importance_values.sort_values(by='Information Gain', ascending=True).reset_index(drop=True)

# Feature Importance Slider 
max_slider_value=max(feature_importance_values['Information Gain'])
values=st.slider('Select a range of values', 0.0, max_slider_value, (0.0, max_slider_value))
st.write('Looking at features with an information gain value between:', values[0], 'and ', round(values[1], 2))

feature_importance_values=feature_importance_values[(feature_importance_values['Information Gain'] >= values[0]) &\
                                                    (feature_importance_values['Information Gain'] <= values[1])]


# Feature Importance Scatter Plot
feature_importance_scatter=px.scatter(feature_importance_values, 
                                      x=feature_importance_values.index, 
                                      y="Information Gain", 
                                      color="Information Gain",
                                      size='Information Gain', 
                                      hover_data=['Feature', 'Information Gain'])
feature_importance_scatter.update_layout(title_text='XGBoost Feature Importance Scatter Plot',
                                         title_x=0.25)


feature_importance_values=feature_importance_values.sort_values(by='Information Gain', ascending=False)
# Feature Importance Table
feature_importance_table=go.Figure(data=[go.Table(header=dict(values=list(feature_importance_values.columns)),
                                                  cells=dict(values=[feature_importance_values['Feature'], 
                                                                     feature_importance_values['Information Gain']]))])
feature_importance_table.update_layout(title_text='XGBoost Feature Importance Table',
                                         title_x=0.0)


container2 = st.container()
col3, col4 = st.columns([3, 1])
with container2:
    with col3:
        st.plotly_chart(feature_importance_scatter, use_container_width=True)
    with col4:
        st.plotly_chart(feature_importance_table, use_container_width=True)