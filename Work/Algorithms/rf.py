from sklearn.ensemble import RandomForestClassifier
def get_rf_model(X_train, X_test, y_train, y_test):
    model_rf = RandomForestClassifier(n_estimators=100)
    model_rf.fit(X_train , y_train)
    return model_rf,model_rf.score(X_test , y_test)