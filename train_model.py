import time
def train_model(X_train, y_train, X_test, y_test, param_range, model_name='SVM'):
    models = []
    scores = []
    durations = []
    for param in param_range:
        if model_name == 'kNN':
            print('kNN£¨k={}£©...'.format(param))
            model = KNeighborsClassifier(n_neighbors=param)
        elif model_name == 'LR':
            print('Logistic Regression£¨C={}£©...'.format(param))
            model = LogisticRegression(C=param)
        elif model_name == 'SVM':
            print('SVM£¨C={}£©...'.format(param))
            model = SVC(kernel='linear', C=param)
        start = time.time()
        model.fit(X_train, y_train)
        end = time.time()
        duration = end - start
        print('used_time{:.4f}s'.format(duration))
        score = model.score(X_test, y_test)
        print('acc_ratio£º{:.3f}'.format(score))
        models.append(model)
        durations.append(duration)
        scores.append(score)
    mean_duration = np.mean(durations)
    print('mean_used_time{:.4f}s'.format(mean_duration))
    print()
    best_idx = np.argmax(scores)
    best_acc = scores[best_idx]
    best_model = models[best_idx]
    return best_model, best_acc, mean_duration