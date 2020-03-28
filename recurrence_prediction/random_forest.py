import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, recall_score, precision_score
)
from sklearn.model_selection import GridSearchCV

n_estimators = [50, 100, 150, 200, 300]
max_depth = [3, 5, 15, 25]


def evaluate(df_train, df_test, features, title=None):
    """
    Given training and testing datasets and a list of features,
    trains and evaluates a random forest classifier

    :param df_train: train dataset
    :param df_test: test dataset
    :param features: features to use in the classifer
    :param title: (optional) title (for figures)
    """
    X = df_train[features]
    Y = df_train.recurrence

    forest = RandomForestClassifier(random_state=0)
    hyperF = dict(n_estimators=n_estimators, max_depth=max_depth)

    grid_clf = GridSearchCV(forest, hyperF, cv=5, n_jobs=-1)
    clf = grid_clf.fit(X, Y)

    importances = clf.best_estimator_.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    sorted_importances = [importances[i] for i in sorted_indices]

    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    x = range(1, len(sorted_indices)+1)
    ax.bar(x, sorted_importances)
    if title:
        ax.set_title(f"Feature Importances for {title}")
    ax.set_ylabel('Relative Importance')
    ax.set_xticks(range(1, len(sorted_indices)+1))
    ax.set_xticklabels([X.columns[i] for i in sorted_indices])
    plt.show()

    y_pred = clf.predict_proba(df_test[features])[:, 1]

    y_test = df_test.recurrence
    print(f'Overall Accuracy: {accuracy_score(y_test, np.round(y_pred))}')
    print(f'Overall AUC: {roc_auc_score(y_test, y_pred)}')
    print(f'Overall Recall: {recall_score(y_test, np.round(y_pred))}')
    print(f'Overall Precision: {precision_score(y_test, np.round(y_pred))}')

    df_s1 = df_test[np.logical_or(df_test['stage 1b'], df_test['stage 1a'])]
    y_test1 = df_s1.recurrence
    y_pred1 = clf.predict_proba(df_s1[features])[:, 1]
    print(f'\nStage 1 Accuracy: {accuracy_score(y_test1, np.round(y_pred1))}')
    print(f'Stage 1 Recall: {recall_score(y_test1, np.round(y_pred1))}')

    df_s2 = df_test[np.logical_or(df_test['stage 2b'], df_test['stage 2a'])]
    y_test2 = df_s2.recurrence
    y_pred2 = clf.predict_proba(df_s2[features])[:, 1]
    print(f'\nStage 2 Accuracy: {accuracy_score(y_test2, np.round(y_pred2))}')
    print(f'Stage 2 Precision: {precision_score(y_test2, np.round(y_pred2))}')
