from sklearn.svm import SVC

def get_svm_model(X_train, X_test, y_train, y_test):
    model_svm = SVC(kernel='rbf')
    model_svm.fit(X_train , y_train)  
    return model_svm , model_svm.score(X_test , y_test)