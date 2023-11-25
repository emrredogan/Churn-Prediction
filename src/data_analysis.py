import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 

class DataAnalysis:

    def categorize_columns(data):
        '''
        categorize columns into numeric, categoric and date types
        input: 
            data
        output:
            three lists
        '''
        print('Starting categorize columns...')
        numeric_cols = []
        categoric_cols = []
        date_cols = []

        for column in data.columns:
            if pd.api.types.is_numeric_dtype(data[column]):
                numeric_cols.append(column)
            elif pd.api.types.is_string_dtype(data[column]):
                categoric_cols.append(column)
            else:
                date_cols.append(column)

        return numeric_cols, categoric_cols, date_cols
    

    def check_dtype_cols(data, col_types_list):
        '''
        Check data types of each categorize list.
        '''
        for column in data.columns:
            if column in col_types_list:
                print(f"{column} type is {data[column].dtype}") 
                print(f"Check the first 3 rows of the column: \n{data[data[column].notnull()][:3][column]}")
            else:
                pass 

    def descriptive_stats_for_numeric_cols(data):
        '''
        Calculate descriptive stats for each numeric feature.
        '''
        result_df = pd.DataFrame(columns=['Feature_Name', 'Count', 'Missing_Count', 'Missing_Percentage' ,'Min', 'Max',\
                                          'Std', '25P', '50P', '75P', '95P'])

        for column in data.columns:
            count_ = data[column].count()
            missing_count = data[column].isna().sum()
            missing_percentage = missing_count / len(data[column])
            min_value = data[column].min()
            max_value= data[column].max()
            std_value = data[column].std()
            p25 = data[column].quantile(0.25)
            p50 = data[column].quantile(0.50)
            p75 = data[column].quantile(0.75)
            p95 = data[column].quantile(0.95)

            result_df = result_df.append({
                'Feature_Name': column,
                'Count': count_,
                'Missing_Count': missing_count,
                'Missing_Percentage': missing_percentage,
                'Min': min_value,
                'Max': max_value,
                'Std': std_value,
                '25P': p25,
                '50P': p50,
                '75P': p75,
                '95P': p95,
            }, ignore_index=True)
        
        return result_df

    
    def description_for_categoric_cols(data):
        '''
        Calculate description points for categoric features
        '''
        result_df = pd.DataFrame(columns=['Feature_Name', 'Count', 'Missing_Count', 'Missing_Percentage', 'Value_Counts', \
                                          'Number_Of_Unique', 'Mode'])
        
        for column in data.columns:
            count_ = data[column].count()
            missing_count = data[column].isna().sum()
            missing_percentage = missing_count / len(data[column])
            value_counts = data[column].value_counts()
            unique_values = data[column].nunique()
            mode_ = data[column].mode().values[0]

            result_df = result_df.append({
                'Feature_Name': column,
                'Count': count_,
                'Missing_Count': missing_count,
                'Missing_Percentage': missing_percentage,
                'Value_Counts': value_counts,
                'Number_Of_Unique': unique_values,
                'Mode': mode_,
            }, ignore_index=True)

        return result_df
    

    def plot_histogram(data, feature_name, bins=20):
        plt.hist(data[feature_name], bins=bins)
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of {feature_name}')
        plt.show()

    
    def plot_kde(data, feature_name):
        sns.kdeplot(data[feature_name], shade=True)
        plt.xlabel('Values')
        plt.ylabel('Density')
        plt.title(f'KDE Plot of {feature_name}')
        plt.show()


    def plot_boxplot(data, feature_name):
        sns.boxplot(x=data[feature_name])
        plt.xlabel(f'{feature_name}')
        plt.title(f'Box Plot of {feature_name}')
        plt.show()

    def plot_scatter(data, feature_name1, feature_name2):
        plt.scatter(data[feature_name1], data[feature_name2])
        plt.xlabel(feature_name1)
        plt.ylabel(feature_name2)
        plt.title(f'Scatter Plot of {feature_name1} vs {feature_name2}')
        plt.show()
        
    def bar_plot_categoric_feat(data, feature_name):
        category_counts = data[feature_name].value_counts()
        plt.figure(figsize=(8,6))
        category_counts.plot(kind='bar', color='skyblue')
        plt.title(f'Category Counts for {feature_name}')
        plt.xlabel(f'Categories for {feature_name}')
        plt.ylabel('Counts')
        plt.xticks(rotation=0)
        plt.show()