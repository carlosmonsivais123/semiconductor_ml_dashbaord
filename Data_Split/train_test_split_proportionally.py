import pandas as pd
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Create_Train_Test_Split:
    def create_and_save_train_test_split(self, df, random_state_value, training_data_output_location, testing_data_output_location):
        df['Label']=df['Label'].astype(str)
        train, test=train_test_split(df, 
                                     test_size=0.2, 
                                     stratify=df[['Label', 'day_of_week']],
                                     random_state=random_state_value,
                                     shuffle=True)
        
        train.to_csv(training_data_output_location, header=True, index=False)
        test.to_csv(testing_data_output_location, header=True, index=False)

        print(f'Training data has been output to {training_data_output_location}')
        print(f'Testing data has been output to {testing_data_output_location}\n')

        return train, test
    

    def evaluate_data_balance_for_train_test_split(self, train_df, test_df, plot_location):
        train_label_counts=train_df.groupby('Label').count()['Time'].reset_index(drop=False).rename(columns={'Time': 'Count'})
        train_label_counts['Percentage']=(train_label_counts['Count']/train_label_counts['Count'].sum()) * 100

        test_label_counts=test_df.groupby('Label').count()['Time'].reset_index(drop=False).rename(columns={'Time': 'Count'})
        test_label_counts['Percentage']=(test_label_counts['Count']/test_label_counts['Count'].sum()) * 100

        train_labels_by_dow=train_df.groupby(['day_of_week', 'Label'])['Time'].count().reset_index(drop=False)
        train_labels_by_dow=train_labels_by_dow.rename(columns={'Time': 'Count'})
        train_labels_by_dow['Percentage']=(train_labels_by_dow['Count']/train_labels_by_dow['Count'].sum()) * 100

        test_labels_by_dow=test_df.groupby(['day_of_week', 'Label'])['Time'].count().reset_index(drop=False)
        test_labels_by_dow=test_labels_by_dow.rename(columns={'Time': 'Count'})
        test_labels_by_dow['Percentage']=(test_labels_by_dow['Count']/test_labels_by_dow['Count'].sum()) * 100

        fig = make_subplots(rows=3, cols=2,
                            subplot_titles=('Training Data Label Proportions', 
                                            'Testing Data Label Proportions', 
                                            'Training Data Label Proportions by Day of Week: Label 1', 
                                            'Testing Data Label Proportions by Day of Week: Label 1',
                                            'Training Data Label Proportions by Day of Week: Label -1',
                                            'Testing Data Label Proportions by Day of Week: Label -1'),
                            shared_yaxes='rows')

        fig.add_trace(go.Bar(x=train_label_counts['Label'], 
                             y=train_label_counts['Percentage'], 
                             name='Training Data Label Proportions',
                             text=round(train_label_counts['Percentage'], 2),
                             textposition='auto'),
                      row=1, col=1)

        fig.add_trace(go.Bar(x=test_label_counts['Label'], 
                             y=test_label_counts['Percentage'], 
                             name='Testing Data Label Proportions',
                             text=round(test_label_counts['Percentage'], 2),
                             textposition='auto'),
                      row=1, col=2)

        fig.add_trace(go.Bar(x=train_labels_by_dow[train_labels_by_dow['Label']=='1']['day_of_week'], 
                             y=train_labels_by_dow[train_labels_by_dow['Label']=='1']['Percentage'], 
                             name='Training Data Label Proportions by Day of Week: Label 1',
                             text=round(train_labels_by_dow[train_labels_by_dow['Label']=='1']['Percentage'], 2),
                             textposition='auto'),
                    row=2, col=1)

        fig.add_trace(go.Bar(x=test_labels_by_dow[test_labels_by_dow['Label']=='1']['day_of_week'], 
                             y=test_labels_by_dow[test_labels_by_dow['Label']=='1']['Percentage'], 
                             name='Testing Data Label Proportions by Day of Week: Label 1',
                             text=round(test_labels_by_dow[test_labels_by_dow['Label']=='1']['Percentage'], 2),
                             textposition='auto'),
                    row=2, col=2)
        

        fig.add_trace(go.Bar(x=train_labels_by_dow[train_labels_by_dow['Label']=='-1']['day_of_week'], 
                             y=train_labels_by_dow[train_labels_by_dow['Label']=='-1']['Percentage'], 
                             name='Training Data Label Proportions by Day of Week: Label -1',
                             text=round(train_labels_by_dow[train_labels_by_dow['Label']=='-1']['Percentage'], 2),
                             textposition='auto'),
                    row=3, col=1)


        fig.add_trace(go.Bar(x=test_labels_by_dow[test_labels_by_dow['Label']=='-1']['day_of_week'], 
                             y=test_labels_by_dow[test_labels_by_dow['Label']=='-1']['Percentage'], 
                             name='Testing Data Label Proportions by Day of Week: Label -1',
                             text=round(test_labels_by_dow[test_labels_by_dow['Label']=='-1']['Percentage'], 2),
                             textposition='auto'),
                    row=3, col=2)
        
        
        fig.update_layout(showlegend=False, 
                          title_text="Training and Testing Data Label Proportions",
                          title_x=0.5,
                          height=900, 
                          width=1300)

        fig.write_image(f"{plot_location}/train_test_label_proportions.png")