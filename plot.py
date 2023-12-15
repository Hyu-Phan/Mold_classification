from matplotlib import pyplot as plt
import seaborn as sns


def plot_data(df):
    label_count = df['label'].value_counts()
    # plot label distribution
    plt.figure(figsize=(10, 7))
    sns.barplot(x=label_count.index, y=label_count)
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.xticks(rotation=90, ha='right')

    plt.subplots_adjust(bottom=0.3)

    # Thêm số lượng vào mỗi cột
    for i, v in enumerate(label_count):
        plt.text(i, v, str(v), ha='center', va='bottom')
    plt.show()


def plot_score(results, features):
    # Tạo danh sách các môi trường (env)
    envs = [result['env'] for result in results]

    # Tạo danh sách các giá trị accuracy, precision, recall, f1 tương ứng cho mỗi môi trường
    accuracy_values = [result['accuracy'] for result in results]
    precision_values = [result['precision'] for result in results]
    recall_values = [result['recall'] for result in results]
    f1_values = [result['f1'] for result in results]

    # Tạo biểu đồ cột
    x = range(len(envs))
    width = 0.2

    fig, ax = plt.subplots()
    rects1 = ax.bar(x, accuracy_values, width, label='Accuracy')
    rects2 = ax.bar([i + width for i in x], precision_values,
                    width, label='Precision')
    rects3 = ax.bar([i + 2 * width for i in x],
                    recall_values, width, label='Recall')
    rects4 = ax.bar([i + 3 * width for i in x], f1_values, width, label='F1')

    # Thêm giá trị lên trên mỗi cột
    for rects in [rects1, rects2, rects3, rects4]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3),
                        textcoords="offset points", ha='center', va='bottom')

    # Đặt các nhãn và tiêu đề cho biểu đồ
    ax.set_ylabel('Scores')
    ax.set_title(f'Evaluation Scores by Environment and Feature ({features})')
    ax.set_xticks([i + 1.5 * width for i in x])
    ax.set_xticklabels(envs)
    ax.legend()

    # Hiển thị biểu đồ
    plt.show()


def plot_auc():
    pass
    # Xác suất dự đoán cho nhãn positive (label=1)
    # y_pred_ds = ds_clf.predict_proba(X_test)
    # y_test_bin = label_binarize(y_test, classes=ds_clf.classes_)
    # fpr_ds, tpr_ds, _ = roc_curve(y_test_bin.ravel(), y_pred_ds.ravel())
    # roc_auc_ds = auc(fpr_ds, tpr_ds)

    # rf_clf.fit(X_train, y_train)
    # score = rf_clf.score(X_test, y_test)
    # y_pred_rf = rf_clf.predict_proba(X_test)
    # fpr_rf, tpr_rf, _ = roc_curve(y_test_bin.ravel(), y_pred_rf.ravel())
    # roc_auc_rf = auc(fpr_rf, tpr_rf)

    # plt.figure()
    # plt.plot(fpr_ds, tpr_ds, label=f'Decision Tree (AUC = {roc_auc_ds:.2f})')
    # plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
    # plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random Guess')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Compare ROC')
    # plt.legend()
    # plt.show()
    # print('Random Forest Classifier: ', score)
