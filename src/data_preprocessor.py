import pandas as pd
from sklearn.impute import  SimpleImputer, IterativeImputer
from sklearn.preprocessing import RobustScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import  roc_auc_score
import xgboost as xgb 

class Preprocessor:

    def check_duplicate_cols(data):
        duplicate_cols = data.T.duplicated()
        duplicate_columns = data.columns[duplicate_cols].tolist()
        return duplicate_columns
    
    def check_duplicate_rows(data):
        duplicate_rows = data[data.duplicated()]
        return duplicate_rows
    

    def simple_imputer(data, strategy):
        imputer = SimpleImputer(strategy=strategy)
        data_imputed = imputer.fit_transform(data)
        return data_imputed

    def iterative_imputer(data, max_iter=10):
        imputer = IterativeImputer(max_iter=max_iter, random_state=0)
        data_imputed = imputer.fit_transform(data)
        return data_imputed

    def standard_scaler(data):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        return scaled_data
    
    def robust_scaler(data):
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(data)
        return scaled_data
    
    def label_encoder(data, column_name):
        label_encoder = LabelEncoder()
        data[column_name + '_encoded'] = label_encoder.fit_transform(data[column_name])
        return data 
    
    def one_hot_encoder(data, column_name):
        ohe = pd.get_dummies(data[column_name], prefix=column_name)
        data = pd.concat([data, ohe], axis=1)
        return data 
    

    def split_data(X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
        return X_train, X_test, y_train, y_test
    
    def calculate_univariate_gini(data, target_name, X_test, y_test):
        univariate_gini = {}

        for feature in data.columns:
            if feature != target_name:
                X = data[[feature]]
                y = data[target_name]

                model = xgb.XGBClassifier()
                model.fit(X,y)

                y_pred = model.predict_proba(X_test[feature])[:,1]
                roc_auc = roc_auc_score(y_test, y_pred)
                gini = 2 * roc_auc - 1 

                univariate_gini[feature] = gini 
        
        return univariate_gini
    