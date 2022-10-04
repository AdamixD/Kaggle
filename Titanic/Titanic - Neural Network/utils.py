import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def repair_data(data):
    data1 = repair_numeric_data(data, 'Age')
    data2 = repair_numeric_data(data1, 'Fare')
    data3 = repair_object_data(data2, 'Embarked')
    return data3


def repair_numeric_data(data, col_name):
    median = data[col_name].median()
    data[col_name].fillna(median, inplace=True)
    return data


def repair_object_data(data, col_name):
    val = data[col_name].value_counts().idxmax()
    data[col_name].fillna(val, inplace=True)
    return data


def add_new_features(data):
    new_data1 = add_family_members_feature(data)
    new_data2 = add_alone_feature(new_data1)
    return new_data2


def add_family_members_feature(data):
    new_data = data['SibSp'] + data['Parch']
    data.insert(8, "Family Members", new_data, True)
    return data


def add_alone_feature(data):
    temporary_data = data["Family Members"]
    data.insert(9, "Not Alone", temporary_data, True)
    data.loc[data["Not Alone"] != 0, "Not Alone"] = 1
    return data


def convert_cat_to_num(data):
    for col_name in data.columns:
        if data[col_name].dtype == 'object':
            data[col_name] = pd.factorize(data[col_name])[0]
    return data


def get_clear_prediction(model, X):
    y_predicted = model.predict(X)
    for i in range(len(y_predicted)):
        y_predicted[i, 0] = 1 if y_predicted[i, 0] > 0.5 else 0
    return y_predicted


def find_errors(model, X, y):
    y_predicted = np.squeeze(np.asarray(get_clear_prediction(model, X)))
    y = np.squeeze(np.asarray(y))
    errors_idxs = np.where(y_predicted != y)[0]
    return errors_idxs


def compute_accuracy(model, X, y):
    accuracy = (X.shape[0] - len(find_errors(model, X, y))) / X.shape[0]
    print('Accuracy: ', accuracy)
    return accuracy


def widgvis(fig):
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False


def plt_loss_tf(history):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    widgvis(fig)
    ax.plot(history.history['loss'], label='loss')
    ax.set_ylim([0, 2])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('loss (cost)')
    ax.legend()
    ax.grid(True)
    plt.show()
