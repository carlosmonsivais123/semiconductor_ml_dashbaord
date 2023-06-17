import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTENC
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix 
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

class Create_Classification_Models:
    def __init__(self, random_state):
        self.random_state=random_state

    def create_X_and_y_data(self, train_df, test_df):
        X_train=train_df[[ele for ele in train_df.columns.tolist() if ele not in ['Label', 'Time']]]
        X_train.columns=X_train.columns.astype(str)
        y_train=pd.DataFrame(train_df['Label'])
        y_train['Label']=y_train['Label'].map({"-1": 0, "1": 1})

        X_test=test_df[[ele for ele in test_df.columns.tolist() if ele not in ['Label', 'Time']]]
        X_test.columns=X_test.columns.astype(str)
        y_test=pd.DataFrame(test_df['Label'])
        y_test['Label']=y_test['Label'].map({"-1": 0, "1": 1})

        return X_train, y_train, X_test, y_test
    

    def smote_categorical_technique(self, train_df, test_df):
        oversmote=SMOTENC(categorical_features=[145], 
                          random_state=self.random_state)
        train_df, test_df=oversmote.fit_resample(train_df, test_df)

        return train_df, test_df


    def data_scaling_and_encoding(self, df):
        numerical_values=[ele for ele in df.columns.tolist() if ele not in ['day_of_week']]
        categorical_values=[df.columns.tolist().pop()]

        numerical_pipeline=Pipeline([('scale', StandardScaler())])
        categorical_pipeline=Pipeline([('one_hot_encoding', OneHotEncoder(handle_unknown='ignore'))])

        col_transformer=ColumnTransformer(transformers=[('num_pipeline', numerical_pipeline, numerical_values),
                                                        ('cat_pipeline', categorical_pipeline, categorical_values)],
                                                        remainder='passthrough')

        return col_transformer


    def gridsearch_cv_best_model_mlflow(self, train_df, test_df, X_test, y_test):
        train_df, test_df=self.smote_categorical_technique(train_df=train_df, 
                                                           test_df=test_df)
        
        model_selection=['xgboost', 
                         'logistic_regression', 
                         'random_forest', 
                         'support_vector_machine', 
                         'naive_bayes', 
                         'k_nearest_neighbor']
        
        model_dictionary={'xgboost': xgb.XGBClassifier(random_state=self.random_state),
                          'logistic_regression': LogisticRegression(random_state=self.random_state),
                          'random_forest': RandomForestClassifier(random_state=self.random_state),
                          'support_vector_machine': SVC(random_state=self.random_state),
                          'naive_bayes': GaussianNB(),
                          'k_nearest_neighbor': KNeighborsClassifier()}
        
        model_parameter_dictionary={'xgboost': [{'objective': ['binary:logistic', 'reg:logistic'],
                                                 'max_depth': [6, 20, 40],
                                                 'n_estimators': [100, 200, 300, 400],
                                                 'booster': ['gbtree', 'gblinear', 'dart'], 
                                                 'learning_rate': [0.01, 0.1, 1.0]}], 
                                    'logistic_regression': [{'C': [0.01, 0.1, 1.0],
                                                             'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
                                                             'max_iter': [100, 200]}], 
                                    'random_forest': [{'n_estimators': [100, 150, 200],
                                                       'criterion': ['gini', 'entropy', 'log_loss']}], 
                                    'support_vector_machine': [{'C': [0.001, 0.1, 1.0],
                                                                'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}], 
                                    'naive_bayes': [{'var_smoothing': [0.001, 0.000001, 0.000000001]}], 
                                    'k_nearest_neighbor': [{'n_neighbors': [2, 3, 4],
                                                             'weights': ['uniform', 'distance'],
                                                             'algorithm': ['ball_tree', 'kd_tree', 'brute']}]}

        for model in model_selection:
            with mlflow.start_run(run_name=f'{model}_gridsearch_cv_smote'):
                column_transformer=self.data_scaling_and_encoding(df=train_df)

                grid_search_optimization=GridSearchCV(model_dictionary[model],
                                                    param_grid=model_parameter_dictionary[model],
                                                    scoring=accuracy_score,
                                                    refit=True,
                                                    cv=3)

                sklearn_pipeline=Pipeline(steps=[('col_transformer_step', column_transformer),
                                                (f'{model}_cv_step', grid_search_optimization)])

                sklearn_pipeline.fit(train_df, test_df)

                predictions=sklearn_pipeline.predict(X_test)

                mlflow.log_metric(f"accuracy", 
                                  accuracy_score(y_test, predictions))
                mlflow.log_metric(f"precision", 
                                  precision_score(y_test, predictions))
                mlflow.log_metric(f"recall", 
                                  recall_score(y_test, predictions))
                mlflow.log_metric(f"f1", 
                                  f1_score(y_test, predictions))
                mlflow.log_dict(confusion_matrix(y_test, predictions).tolist(), 
                                f"confusion_matrix.txt")

                signature=infer_signature(X_test, predictions)
                mlflow.sklearn.log_model(sklearn_pipeline, 
                                         f"{model}_smote", 
                                         signature=signature, 
                                         registered_model_name=f"{model}_smote")

                run = mlflow.active_run()
                print("Run ID: {}".format(run.info.run_id))