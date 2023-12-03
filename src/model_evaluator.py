import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import shap 


def ModelEvaluator(classifier,x_test,y_test):
    
    cm = confusion_matrix(y_test,classifier.predict(x_test))
    names = ['True Neg','False Pos','False Neg','True Pos']
    counts = [value for value in cm.flatten()]
    percentages = ['{0:.3%}'.format(value) for value in cm.flatten()/np.sum(cm)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names,counts,percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cm,annot = labels,cmap ='viridis',fmt ='')
    
    print(classification_report(y_test,classifier.predict(x_test)))

def plot_feature_importance(model, feature_names):
    if isinstance(model, xgb.XGBClassifier):
        importance = model.feature_importances_
    elif isinstance(model, RandomForestClassifier):
        importance = model.feature_importances_
    else:
        raise ValueError("Model type not supported for feature importance.")
    
    indices = np.argsort(importance)[::-1]
    sorted_names = [feature_names[i] for i in indices]
    sorted_importance = importance[indices]

    plt.figure(figsize=(10,6))
    plt.bar(range(len(importance)), sorted_importance, align='center')
    plt.xticks(range(len(importance)), sorted_names, rotation=90)
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()


def shap_analysis(model, X_train, X_test):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    shap.initjs()
    #summary plot
    shap.summary_plot(shap_values[1], X_test)

    