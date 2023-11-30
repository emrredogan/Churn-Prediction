import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import  SimpleImputer, IterativeImputer
from sklearn.preprocessing import RobustScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import  roc_auc_score
import xgboost as xgb 
import matplotlib.pyplot as plt
import seaborn as sns 

class Preprocessor:

    def check_duplicate_cols(data):
        '''
        Check duplicate columns
        '''
        duplicate_cols = data.T.duplicated()
        duplicate_columns = data.columns[duplicate_cols].tolist()
        return duplicate_columns
    
    def check_duplicate_rows(data):
        '''
        Check duplicade rows 
        '''
        duplicate_rows = data[data.duplicated()]
        return duplicate_rows
    
    def return_only_missing_counts(data):
        for column in data.columns:
            if data[column].isna().sum() != 0:
                print(f"{column} has {data[column].isna().sum()} missing values.")
            else:
                #print(f"{column} has not missing value.")
                pass


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
        data[column_name] = label_encoder.fit_transform(data[column_name])
        return data 
    
    def one_hot_encoder(data, column_name):
        ohe = pd.get_dummies(data[column_name], prefix=column_name)
        data = pd.concat([data, ohe], axis=1)
        del data[column_name]
        return data 
    

    def split_data(X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
        return X_train, X_test, y_train, y_test
    
    def calculate_univariate_gini(data, target_name, X_test, y_test):
        univariate_gini = []

        for feature in data.columns:
            if feature != target_name:
                X = data[[feature]]
                y = data[target_name]

                model = xgb.XGBClassifier()
                model.fit(X,y)

                y_pred = model.predict_proba(X_test[feature])[:,1]
                roc_auc = roc_auc_score(y_test, y_pred)
                gini = 2 * roc_auc - 1 

                univariate_gini.append((feature,gini))
        
        return pd.DataFrame(univariate_gini, columns=['Feature_Name', 'Gini_Score'])
    

    def corr_heatmap(data):
        plt.figure(figsize=(20,15))
        sns.heatmap(data.corr(), annot=True, cmap=plt.cm.PuBu)
        plt.show()

    
    def correlation_elimination(data, gini_df, threshold=0.70):
        '''
        Remove highly correlated feature (above threshold) that has lower correlation with target.
        '''
        correlation_matrix = data.corr()
        highly_correlated = correlation_matrix[(correlation_matrix > threshold) & (correlation_matrix < 1.0)].stack().reset_index()
        highly_correlated.columns = ['Variable_1', 'Variable_2', 'Correlation']

        keep_dict = {}
        drop_dict = {}

        for index, row in highly_correlated.iterrows():
            var1 = row['Variable_1']
            var2 = row['Variable_2']
    
            if var1 in gini_df['Feature_Name'].values and var2 in gini_df['Feature_Name'].values:
                gini_var1 = gini_df.loc[gini_df['Feature_Name'] == var1, 'Gini_Score'].values[0]
                gini_var2 = gini_df.loc[gini_df['Feature_Name'] == var2, 'Gini_Score'].values[0]
        
                if gini_var1 >= gini_var2:
                    keep_dict[var1] = gini_var1
                    drop_dict[var2] = gini_var2
                else:
                    keep_dict[var2] = gini_var2
                    drop_dict[var1] = gini_var1

        keep_list = list(keep_dict.keys())
        drop_list = list(drop_dict.keys())

        return drop_list, keep_list


