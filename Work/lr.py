from sklearn.linear_model import LogisticRegression
def get_lr_model(X_train, X_test, y_train, y_test):
    model_lr = LogisticRegression()
    model_lr.fit(X_train , y_train)  
    return model_lr , model_lr.score(X_test , y_test)