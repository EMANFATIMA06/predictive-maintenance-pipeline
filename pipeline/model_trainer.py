from xgboost import XGBClassifier

def train_model(X_train, y_train):
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale = neg / pos
    model = XGBClassifier(
        n_estimators=100,
        scale_pos_weight=scale,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    return model