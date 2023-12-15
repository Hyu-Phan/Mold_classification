from collections import defaultdict
from unittest import result
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, f1_score, make_scorer, accuracy_score, precision_score, recall_score, roc_curve, auc
from sklearn.preprocessing import label_binarize, MinMaxScaler, StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import randint
from itertools import product
import math
from plot import *


def read_and_split_data(i, features=['full'], env=None):
    env_name = '' if env == None else f'_{env}'
    data = pd.DataFrame()
    label = None
    for feature in features:
        path = f'feature2/{feature}/feature_{feature}{env_name}.csv'
        df = pd.read_csv(path)
        label = df['label']
        df = df.drop(['label'], axis=1)
        data = pd.concat([data, df], axis=1)
    data['label'] = label
    label_count = data['label'].value_counts()

    train_data = []
    test_data = []
    rate_train = 0.8 if env == None else 0.6
    random_states = [21, 42, 63, 84]
    for label, count in label_count.items():
        train_count = int(np.ceil(count * rate_train))
        test_count = count - train_count
        label_data = data[data['label'] == label]
        train_subset = label_data.sample(
            n=train_count, random_state=random_states[i])
        test_subset = label_data.drop(train_subset.index)
        train_data.append(train_subset)
        test_data.append(test_subset)

    train_data = pd.concat(train_data)
    test_data = pd.concat(test_data)
    print(train_data.shape)
    print(test_data.shape)
    return train_data, test_data


def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    return score, precision, recall, f1


def train_model(train_data, test_data):
    X_train = train_data.drop(['label'], axis=1)
    y_train = train_data['label']
    X_test = test_data.drop(['label'], axis=1)
    y_test = test_data['label']
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    knn_clf = KNeighborsClassifier(n_neighbors=3)

    svm_clf = SVC(kernel='rbf', C=1, gamma=0.1, probability=True)

    ds_clf = DecisionTreeClassifier(criterion='entropy',
                                    max_depth=50,
                                    max_leaf_nodes=100,
                                    )

    nn_clf = MLPClassifier(hidden_layer_sizes=(100, 100),
                           activation='relu',
                           solver='adam',
                           alpha=0.0001,
                           batch_size='auto',
                           learning_rate='constant',
                           learning_rate_init=0.001,
                           power_t=0.5,
                           max_iter=200,
                           shuffle=True,
                           random_state=None,
                           tol=0.0001,
                           verbose=False,
                           warm_start=False,
                           momentum=0.9,
                           nesterovs_momentum=True,
                           early_stopping=False,
                           validation_fraction=0.1,
                           beta_1=0.9,
                           beta_2=0.999,
                           epsilon=1e-08,
                           n_iter_no_change=10,
                           max_fun=15000,
                           )

    rf_clf = RandomForestClassifier(n_estimators=100,
                                    criterion='entropy',
                                    random_state=21,
                                    max_depth=100,
                                    )

    list_model = [knn_clf, nn_clf, rf_clf]
    list_name = ['KNN', 'SVM', 'Decision Tree', 'MLP', 'Random Forest']

    results = {}
    for model, name in zip(list_model, list_name):
        result = evaluate_model(model, X_train, y_train, X_test, y_test)
        results[name] = result

    return results


def parameter_tunning(train_data, test_data):
    X_train = train_data.drop(['label'], axis=1)
    y_train = train_data['label']
    X_test = test_data.drop(['label'], axis=1)
    y_test = test_data['label']

    # model = RandomForestClassifier()
    # pram_grid = [
    #     {
    #         # {'n_estimators': [50, 100, 150, 200],
    #         #  'max_depth': [20, 30, 50, 100],
    #         #  'max_leaf_nodes': [20, 30],
    #         #  'criterion': ['gini', 'entropy'],
    #         'ccp_alpha': [0.0, 0.1, 0.2]
    #     }
    # ]

    model = DecisionTreeClassifier(random_state=42)
    pram_grid = [
        {
            'criterion': ['gini', 'entropy'],
            'max_depth': list(np.arange(10, 100, step=10)) + [None],
            # 'max_leaf_nodes': [30, 50, 100],
            'ccp_alpha': [0.0, 0.1]
        }
    ]
    score = make_scorer(f1_score, average='macro')
    scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score, average='macro', zero_division=0),
               'recall': make_scorer(recall_score, average='macro', zero_division=0),
               'f1_macro': make_scorer(f1_score, average='macro'),
               'f1_weighted': make_scorer(f1_score, average='weighted')}

    grid_search = GridSearchCV(
        model, pram_grid, cv=4, scoring=scoring, refit='accuracy')
    grid_search.fit(X_train, y_train)
    print("F1-Score:", grid_search.best_params_)
    print("F1-Score:", grid_search.best_score_)
    # print all param and score
    # result = pd.DataFrame(grid_search.cv_results_)
    # result.to_csv('result.csv', index=False)

    pram_dist = {'max_depth': list(np.arange(10, 100, step=10)) + [None],
                 #  'n_estimators': np.arange(10, 500, step=50),
                 'criterion': ['gini', 'entropy'],
                 }

    random_search = RandomizedSearchCV(
        model, pram_dist, cv=4, scoring=scoring, refit='accuracy', n_iter=10)
    random_search.fit(X_train, y_train)
    print("Accuracy:", random_search.best_params_)
    print("Best:", random_search.best_score_)

    # grid_search = GridSearchCV(model, pram_grid, cv=4, scoring=scoring, refit='accuracy')
    # grid_search.fit(X_train, y_train)
    # print("Accuracy:", grid_search.best_params_)
    # print("Accuracy:", grid_search.best_score_)


def main():
    features = ['color', 'glcm', 'lbp', 'gabor', 'texture', 'full']
    envs = ['CYA', 'DG18', 'MEA', 'YES', None]

    results = []
    for feature, env in product(features, envs):
        train_data, test_data = read_and_split_data(feature, env)
        result = train_model(train_data, test_data)
        results.append({
            'feature': feature,
            'env': env,
            'accuracy': result
        })
        # parameter_tunning(train_data, test_data)
        # combine_csv()
    print(results)

    accuracy_by_env = {env: [] for env in envs}

    # Lặp qua kết quả và lưu trữ accuracy theo feature và môi trường
    for result in results:
        feature = result['feature']
        env = result['env']
        accuracy = result['accuracy']
        accuracy_by_env[env].append((feature, accuracy))

    # Plot biểu đồ cho từng môi trường
    for env in envs:
        # Lấy danh sách các feature và accuracy của mỗi feature trong môi trường hiện tại
        data = accuracy_by_env[env]
        features = [item[0] for item in data]
        accuracies = [item[1] for item in data]

        # Tạo biểu đồ
        plt.figure()
        plt.bar(features, accuracies)
        plt.xlabel('Feature')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy of Features in Environment {env}')
        # Đặt giới hạn cho trục y từ 0 đến 1 (hoặc tùy chỉnh theo nhu cầu)
        plt.ylim(0, 1)
        for i in range(len(features)):
            plt.text(
                i, accuracies[i], f'{accuracies[i]:.2f}', ha='center', va='bottom')
        plt.show()


# def cross_validation(envs, cv=3):


def test():
    envs = ['CYA', 'DG18', 'MEA', 'YES']
    features = ['color',
                'glcm',
                'lbp',
                'gabor',
                'glrlm',
                'glszm',
                # 'law'
                ]
    data = defaultdict(list)
    cv = 4
    for env in envs:
        for i in range(cv):
            train, test = read_and_split_data(i, env=env, features=features)
            data[env].append((train, test))

    for i in range(cv):
        train_data = pd.DataFrame()
        test_data = pd.DataFrame()
        for env in envs:
            train, test = data[env][i]
            train_data = pd.concat([train_data, train])
            test_data = pd.concat([test_data, test])
        data['All environment'].append((train_data, test_data))

    results = []
    for env, list_data in data.items():
        score = defaultdict(list)
        for (train, test) in list_data:
            # lọc ra label có số lượng ít hơn 12
            if env == 'All environment':
                label_count = train['label'].value_counts()
                label_count = label_count[label_count >= 12].keys()

                # lọc ra train có label trong label_count
                train = train[train['label'].isin(label_count)]
                test = test[test['label'].isin(label_count)]
            # plot_data(train)
            # plot_data(test)
            result = train_model(train, test)
            for key, value in result.items():
                score[key].append(value)

        for key, value in score.items():
            score[key] = np.mean(value, axis=0)

        # print(f'Average score of {env}: {score}')
    
        results.append({
            'env': env,
            'accuracy': score['MLP'][0],
            'precision': score['MLP'][1],
            'recall': score['MLP'][2],
            'f1': score['MLP'][3]
        })
    plot_score(results, features)


def test_model():
    pass


if __name__ == '__main__':
    # main()
    test()
