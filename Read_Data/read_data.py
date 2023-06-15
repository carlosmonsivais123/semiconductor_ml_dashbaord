import pandas as pd

class Read_And_Merge_Data:
    def __init__(self, secom_data_values_location, secom_data_labels):
        self.secom_data_values_location=secom_data_values_location
        self.secom_data_labels=secom_data_labels


    def read_data(self):
        data_values=pd.read_csv(self.secom_data_values_location, sep=' ', header=None)

        data_labels=pd.read_csv('data/secom_labels.data', sep=' ', header=None)
        data_labels.columns=['Label', 'Time']

        return data_values, data_labels
    

    def merge_values_and_labels(self):
        self.values=self.read_data()[0]
        self.labels=self.read_data()[1]

        print(f'The shape of the values is {self.values.shape}')
        print(f'The shape of the labels is {self.labels.shape}')

        combined_df=pd.concat([self.values, self.labels], axis=1)
        combined_df['Time']=pd.to_datetime(combined_df['Time'], dayfirst=True, format='''%d/%m/%Y %H:%M:%S''')
        combined_df=combined_df.sort_values(by='Time', ascending=True)

        combined_df['day_of_week']=combined_df['Time'].dt.day_name()

        print(f'The shape of the combined values and labels is {combined_df.shape}')

        return combined_df