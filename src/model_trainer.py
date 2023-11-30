from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
import optuna 
import xgboost as xgb


def ModelTrainer(classifier,x_train,y_train,x_test,y_test):
    
    classifier.fit(x_train,y_train)
    pred = classifier.predict(x_test)
    cv = RepeatedStratifiedKFold(n_splits = 10,n_repeats = 3,random_state = 1)
    cr_v_sc = cross_val_score(classifier,x_train,y_train,cv = cv,scoring = 'roc_auc')
    r_a_s = roc_auc_score(y_test,pred)
    gini = 2 * r_a_s - 1 
    print("Cross Validation Score : ",'{0:.2%}'.format(cr_v_sc.mean()))
    print("ROC_AUC Score : ",'{0:.2%}'.format(r_a_s))
    print("Gini Score : ",'{0:.2%}'.format(gini))


def objective(trial, X, y):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 400),
        'max_depth': trial.suggest_int('max_depth', 2, 25),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 6),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 1e-8, 1.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.2, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 1.0),
    }

    model = xgb.XGBClassifier(**param)
    xgb_kfold = RepeatedStratifiedKFold(n_splits = 10,n_repeats = 3,random_state = 1)

    accuracies =[]

    for train_idx, valid_idx in xgb_kfold.split(X, y):
        X_train, X_val = X[train_idx], X[valid_idx]
        y_train, y_val = y[train_idx], y[valid_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        accuracies.append(accuracy)

    return sum(accuracies) / len(accuracies)


def gridsearch(X, y, model):
    params = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [4, 5, 6, 7, 8],
    'criterion': ['gini', 'entropy']
    }

    rskf = RepeatedStratifiedKFold(n_splits = 10,n_repeats = 3,random_state = 1)

    grid_search  =GridSearchCV(estimator=model, 
                               param_grid=params,
                               cv=rskf,
                               scoring='accuracy',n_jobs=-1)
    grid_search.fit(X, y)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    return best_params, best_score 

