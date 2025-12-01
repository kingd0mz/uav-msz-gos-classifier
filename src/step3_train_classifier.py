from sklearn.ensemble import RandomForestClassifier

def train_classifier(X, y):
    clf = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X, y)
    return clf
