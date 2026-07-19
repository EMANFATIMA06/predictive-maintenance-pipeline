def evaluate(model, X_test, y_test, threshold=0.3):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    return pred, proba