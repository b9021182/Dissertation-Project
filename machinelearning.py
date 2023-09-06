import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

legitimate_df = pd.read_csv("structured_data_legit.csv")
phishing_df = pd.read_csv("structured_data_phishing.csv")
df = pd.concat([legitimate_df, phishing_df], axis=0)
df = df.sample(frac=1)
df = df.drop('URL', axis=1)
df = df.drop_duplicates()
X = df.drop('label', axis=1)
Y = df['label']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

svm_model = svm.SVC(C=1.0, kernel='rbf', probability=True)
rf_model = RandomForestClassifier(n_estimators=60, max_depth=10, min_samples_split=5, min_samples_leaf=2)
dt_model = tree.DecisionTreeClassifier(max_depth=10, min_samples_split=5, min_samples_leaf=2)
ab_model = AdaBoostClassifier(base_estimator=None, n_estimators=50)
nb_model = GaussianNB()
nn_model = MLPClassifier(alpha=0.01, max_iter=1000)
kn_model = KNeighborsClassifier(n_neighbors=5)

def calculate_measures(TN, TP, FN, FP):
    model_accuracy = (TP + TN) / (TP + TN + FN + FP)
    model_precision = TP / (TP + FP)
    model_recall = TP / (TP + FN)
    return model_accuracy, model_precision, model_recall

def calculate_roc_auc(model, x_test, y_test):
    predictions = model.predict_proba(x_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, predictions)
    auc_score = roc_auc_score(y_test, predictions)
    return fpr, tpr, auc_score

K = 5
total = X.shape[0]
index = int(total / K)

X_1_test = X.iloc[:index]
X_1_train = X.iloc[:index]
Y_1_test = Y.iloc[:index]
Y_1_train = Y.iloc[:index]

X_2_test = X.iloc[index:index * 2]
X_2_train = X.iloc[np.r_[:index, index * 2:]]
Y_2_test = Y.iloc[index:index * 2]
Y_2_train = Y.iloc[np.r_[:index, index * 2:]]

X_3_test = X.iloc[index * 2:index * 3]
X_3_train = X.iloc[np.r_[:index * 2, index * 3:]]
Y_3_test = Y.iloc[index * 2:index * 3]
Y_3_train = Y.iloc[np.r_[:index * 2, index * 3:]]

X_4_test = X.iloc[index * 3:index * 4]
X_4_train = X.iloc[np.r_[:index * 3, index * 4:]]
Y_4_test = Y.iloc[index * 3:index * 4]
Y_4_train = Y.iloc[np.r_[:index * 3, index * 4:]]

X_5_test = X.iloc[index * 4:]
X_5_train = X.iloc[np.r_[:index * 4]]
Y_5_test = Y.iloc[index * 4:]
Y_5_train = Y.iloc[np.r_[:index * 4]]

X_train_list = [X_1_train, X_2_train, X_3_train, X_4_train, X_5_train]
X_test_list = [X_1_test, X_2_test, X_3_test, X_4_test, X_5_test]
Y_train_list = [Y_1_train, Y_2_train, Y_3_train, Y_4_train, Y_5_train]
Y_test_list = [Y_1_test, Y_2_test, Y_3_test, Y_4_test, Y_5_test]

def calculate_avg_metrics(accuracy_list, precision_list, recall_list):
    accuracy = sum(accuracy_list) / len(accuracy_list)
    precision = sum(precision_list) / len(precision_list)
    recall = sum(recall_list) / len(recall_list)
    return accuracy, precision, recall

ab_accuracy_list, ab_precision_list, ab_recall_list = [], [], []
dt_accuracy_list, dt_precision_list, dt_recall_list = [], [], []
nb_accuracy_list, nb_precision_list, nb_recall_list = [], [], []
rf_accuracy_list, rf_precision_list, rf_recall_list = [], [], []
svm_accuracy_list, svm_precision_list, svm_recall_list = [], [], []
nn_accuracy_list, nn_precision_list, nn_recall_list = [], [], []
kn_accuracy_list, kn_precision_list, kn_recall_list = [], [], []
ab_auc_list, dt_auc_list, nb_auc_list = [], [], []
rf_auc_list, svm_auc_list, nn_auc_list = [], [], []
kn_auc_list = []

for i in range(K):
    # AdaBoost
    ab_model.fit(X_train_list[i], Y_train_list[i])
    ab_predictions = ab_model.predict(X_test_list[i])
    tn, fp, fn, tp = confusion_matrix(y_true=Y_test_list[i], y_pred=ab_predictions).ravel()
    ab_accuracy, ab_precision, ab_recall = calculate_measures(tn, tp, fn, fp)
    ab_accuracy_list.append(ab_accuracy)
    ab_precision_list.append(ab_precision)
    ab_recall_list.append(ab_recall)
    fpr, tpr, auc_score = calculate_roc_auc(ab_model, X_test_list[i], Y_test_list[i])
    ab_auc_list.append(auc_score)

    # Decision Tree
    dt_model.fit(X_train_list[i], Y_train_list[i])
    dt_predictions = dt_model.predict(X_test_list[i])
    tn, fp, fn, tp = confusion_matrix(y_true=Y_test_list[i], y_pred=dt_predictions).ravel()
    dt_accuracy, dt_precision, dt_recall = calculate_measures(tn, tp, fn, fp)
    dt_accuracy_list.append(dt_accuracy)
    dt_precision_list.append(dt_precision)
    dt_recall_list.append(dt_recall)
    fpr, tpr, auc_score = calculate_roc_auc(dt_model, X_test_list[i], Y_test_list[i])
    dt_auc_list.append(auc_score)

    # Naive Bayes
    nb_model.fit(X_train_list[i], Y_train_list[i])
    nb_predictions = nb_model.predict(X_test_list[i])
    tn, fp, fn, tp = confusion_matrix(y_true=Y_test_list[i], y_pred=nb_predictions).ravel()
    nb_accuracy, nb_precision, nb_recall = calculate_measures(tn, tp, fn, fp)
    nb_accuracy_list.append(nb_accuracy)
    nb_precision_list.append(nb_precision)
    nb_recall_list.append(nb_recall)
    fpr, tpr, auc_score = calculate_roc_auc(nb_model, X_test_list[i], Y_test_list[i])
    nb_auc_list.append(auc_score)

    # Random Forest
    rf_model.fit(X_train_list[i], Y_train_list[i])
    rf_predictions = rf_model.predict(X_test_list[i])
    tn, fp, fn, tp = confusion_matrix(y_true=Y_test_list[i], y_pred=rf_predictions).ravel()
    rf_accuracy, rf_precision, rf_recall = calculate_measures(tn, tp, fn, fp)
    rf_accuracy_list.append(rf_accuracy)
    rf_precision_list.append(rf_precision)
    rf_recall_list.append(rf_recall)
    fpr, tpr, auc_score = calculate_roc_auc(rf_model, X_test_list[i], Y_test_list[i])
    rf_auc_list.append(auc_score)

    # SVM
    svm_model.fit(X_train_list[i], Y_train_list[i])
    svm_predictions = svm_model.predict(X_test_list[i])
    tn, fp, fn, tp = confusion_matrix(y_true=Y_test_list[i], y_pred=svm_predictions).ravel()
    svm_accuracy, svm_precision, svm_recall = calculate_measures(tn, tp, fn, fp)
    svm_accuracy_list.append(svm_accuracy)
    svm_precision_list.append(svm_precision)
    svm_recall_list.append(svm_recall)
    fpr, tpr, auc_score = calculate_roc_auc(svm_model, X_test_list[i], Y_test_list[i])
    svm_auc_list.append(auc_score)

    # Neural Network
    nn_model.fit(X_train_list[i], Y_train_list[i])
    nn_predictions = nn_model.predict(X_test_list[i])
    tn, fp, fn, tp = confusion_matrix(y_true=Y_test_list[i], y_pred=nn_predictions).ravel()
    nn_accuracy, nn_precision, nn_recall = calculate_measures(tn, tp, fn, fp)
    nn_accuracy_list.append(nn_accuracy)
    nn_precision_list.append(nn_precision)
    nn_recall_list.append(nn_recall)
    fpr, tpr, auc_score = calculate_roc_auc(nn_model, X_test_list[i], Y_test_list[i])
    nn_auc_list.append(auc_score)

    # K-Nearest Neighbors
    kn_model.fit(X_train_list[i], Y_train_list[i])
    kn_predictions = kn_model.predict(X_test_list[i])
    tn, fp, fn, tp = confusion_matrix(y_true=Y_test_list[i], y_pred=kn_predictions).ravel()
    kn_accuracy, kn_precision, kn_recall = calculate_measures(tn, tp, fn, fp)
    kn_accuracy_list.append(kn_accuracy)
    kn_precision_list.append(kn_precision)
    kn_recall_list.append(kn_recall)
    fpr, tpr, auc_score = calculate_roc_auc(kn_model, X_test_list[i], Y_test_list[i])
    kn_auc_list.append(auc_score)

AB_accuracy, AB_precision, AB_recall = calculate_avg_metrics(ab_accuracy_list, ab_precision_list, ab_recall_list)
AB_auc = np.mean(ab_auc_list)
print("AdaBoost Accuracy =", AB_accuracy)
print("AdaBoost Precision =", AB_precision)
print("AdaBoost Recall =", AB_recall)
print("AdaBoost AUC =", AB_auc)

DT_accuracy, DT_precision, DT_recall = calculate_avg_metrics(dt_accuracy_list, dt_precision_list, dt_recall_list)
DT_auc = np.mean(dt_auc_list)
print("Decision Tree Accuracy =", DT_accuracy)
print("Decision Tree Precision =", DT_precision)
print("Decision Tree Recall =", DT_recall)
print("Decision Tree AUC =", DT_auc)

NB_accuracy, NB_precision, NB_recall = calculate_avg_metrics(nb_accuracy_list, nb_precision_list, nb_recall_list)
NB_auc = np.mean(nb_auc_list)
print("Naive Bayes Accuracy =", NB_accuracy)
print("Naive Bayes Precision =", NB_precision)
print("Naive Bayes Recall =", NB_recall)
print("Naive Bayes AUC =", NB_auc)

RF_accuracy, RF_precision, RF_recall = calculate_avg_metrics(rf_accuracy_list, rf_precision_list, rf_recall_list)
RF_auc = np.mean(rf_auc_list)
print("Random Forest Accuracy =", RF_accuracy)
print("Random Forest Precision =", RF_precision)
print("Random Forest Recall =", RF_recall)
print("Random Forest AUC =", RF_auc)

SVM_accuracy, SVM_precision, SVM_recall = calculate_avg_metrics(svm_accuracy_list, svm_precision_list, svm_recall_list)
SVM_auc = np.mean(svm_auc_list)
print("Support Vector Machine Accuracy =", SVM_accuracy)
print("Support Vector Machine Precision =", SVM_precision)
print("Support Vector Machine Recall =", SVM_recall)
print("Support Vector Machine AUC =", SVM_auc)

NN_accuracy, NN_precision, NN_recall = calculate_avg_metrics(nn_accuracy_list, nn_precision_list, nn_recall_list)
NN_auc = np.mean(nn_auc_list)
print("Neural Network Accuracy =", NN_accuracy)
print("Neural Network Precision =", NN_precision)
print("Neural Network Recall =", NN_recall)
print("Neural Network AUC =", NN_auc)

KN_accuracy, KN_precision, KN_recall = calculate_avg_metrics(kn_accuracy_list, kn_precision_list, kn_recall_list)
KN_auc = np.mean(kn_auc_list)
print("K-Nearest Neighbors Accuracy =", KN_accuracy)
print("K-Nearest Neighbors Precision =", KN_precision)
print("K-Nearest Neighbors Recall =", KN_recall)
print("K-Nearest Neighbors AUC =", KN_auc)

data = {
    'Accuracy': [AB_accuracy, DT_accuracy, NB_accuracy, RF_accuracy, SVM_accuracy, NN_accuracy, KN_accuracy],
    'Precision': [AB_precision, DT_precision, NB_precision, RF_precision, SVM_precision, NN_precision, KN_precision],
    'Recall': [AB_recall, DT_recall, NB_recall, RF_recall, SVM_recall, NN_recall, KN_recall],
    'AUC': [AB_auc, DT_auc, NB_auc, RF_auc, SVM_auc, NN_auc, KN_auc]
}

index = ['AdaBoost', 'Decision Tree', 'Naive Bayes', 'Random Forest', 'Support Vector Machine', 'Neural Network', 'K-Nearest Neighbors']

df_results = pd.DataFrame(data=data, index=index)
ax = df_results.plot.bar(rot=0)

plt.legend()
plt.show()

# Feature Importance Analysis
feature_importance = pd.DataFrame(index=X.columns)

# AdaBoost
ab_model.fit(X, Y)
ab_feature_importance = ab_model.feature_importances_
feature_importance['AdaBoost'] = ab_feature_importance

# Decision Tree
dt_model.fit(X, Y)
dt_feature_importance = dt_model.feature_importances_
feature_importance['Decision Tree'] = dt_feature_importance

# Naive Bayes (No feature importance available)
feature_importance['Naive Bayes'] = np.nan

# Random Forest
rf_model.fit(X, Y)
rf_feature_importance = rf_model.feature_importances_
feature_importance['Random Forest'] = rf_feature_importance

# SVM (No feature importance available)
feature_importance['Support Vector Machine'] = np.nan

# Neural Network (No feature importance available)
feature_importance['Neural Network'] = np.nan

# K-Nearest Neighbors (No feature importance available)
feature_importance['K-Nearest Neighbors'] = np.nan

# Save feature importance to a CSV file
feature_importance.to_csv('feature_importance.csv')

# Create an empty dictionary to store the FPR and TPR for each model
roc_data = {}

# Models to evaluate
models = {
    'AdaBoost': ab_model,
    'Decision Tree': dt_model,
    'Random Forest': rf_model,
    'Support Vector Machine': svm_model,
    'Neural Network': nn_model,
    'K-Nearest Neighbors': kn_model,
}

# Plot ROC curves for each model separately
for model_name, model in models.items():
    plt.figure(figsize=(6, 6))
    fpr, tpr, auc_score = calculate_roc_auc(model, x_test, y_test)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.2f})')
    plt.title(f'ROC Curve for {model_name}')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.legend()
    plt.grid()
    plt.show()






