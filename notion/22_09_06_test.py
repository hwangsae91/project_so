from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

iris = load_iris()
iris_feature = iris.data
iris_label = iris.target


iris_x_train, iris_x_test, iris_y_train, iris_y_test = train_test_split(iris_feature
                                                    , iris_label
                                                    , test_size=0.2
                                                    , random_state=7)
iris_train_max_size = max(len(iris_x_train), len(iris_y_train))

def lms_1_7_svm(iris_x_train, iris_x_test, iris_y_train, iris_y_test):
    svm_model = svm.SVC(max_iter=iris_train_max_size)
    svm_model.fit(iris_x_train, iris_y_train)
    y_pred = svm_model.predict(iris_x_test)

    print(classification_report(iris_y_test, y_pred))

def lms_1_7_sgd(iris_x_train, iris_x_test, iris_y_train, iris_y_test):
    sgd_model = SGDClassifier(max_iter=iris_train_max_size)
    sgd_model.fit(iris_x_train, iris_y_train)
    y_pred = sgd_model.predict(iris_x_test)

    print(classification_report(iris_y_test, y_pred))

def lms_1_7_log_reg(iris_x_train, iris_x_test, iris_y_train, iris_y_test):
    logistic_model = LogisticRegression(max_iter=iris_train_max_size)
    logistic_model.fit(iris_x_train, iris_y_train)
    y_pred = logistic_model.predict(iris_x_test)

    print(classification_report(iris_y_test, y_pred))

digits_data = load_digits()
digits_feature = digits_data.data
digits_label = digits_data.target

digits_x_train, digits_x_test, digits_y_train, digits_y_test = train_test_split(digits_feature
                                                , digits_label
                                                , test_size= 0.2
                                                , random_state=7)

digits_train_max_size = max(len(digits_x_train), len(digits_y_train))

def lms_1_8_decision_tree(digits_x_train, digits_x_test, digits_y_train, digits_y_test):
    decision_t_model = DecisionTreeClassifier(random_state=42)
    decision_t_model.fit(digits_x_train, digits_y_train)
    y_pred = decision_t_model.predict(digits_x_test)

    print(classification_report(digits_y_test, y_pred))