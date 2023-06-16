import xgboost as xgb

class Create_Classification_Models:
    def create_X_and_y_data(self, train_df, test_df):
        X_train=train_df[[ele for ele in train_df.columns.tolist() if ele not in ['Label', 'Time']]]
        y_train=train_df['Label']

        X_test=test_df[[ele for ele in test_df.columns.tolist() if ele not in ['Label', 'Time']]]
        y_test=test_df['Label']

        return X_train, y_train, X_test, y_test


    def create_xgboost_model(self, train_df):


        return None 