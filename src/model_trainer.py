from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score


def ModelTrainer(classifier,x_train,y_train,x_test,y_test):
    
    classifier.fit(x_train,y_train)
    pred = classifier.predict(x_test)
    cv = RepeatedStratifiedKFold(n_splits = 10,n_repeats = 3,random_state = 1)
    cr_v_sc = cross_val_score(classifier,x_train,y_train,cv = cv,scoring = 'roc_auc')
    r_a_s = roc_auc_score(y_test,pred)
    print("Cross Validation Score : ",'{0:.2%}'.format(cr_v_sc.mean()))
    print("ROC_AUC Score : ",'{0:.2%}'.format(r_a_s))