import numpy as np
import time
from scipy.stats import sem
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# The dataset, 'olivetti faces' has 400 faces and 40 people
from sklearn.datasets import fetch_olivetti_faces
import matplotlib
import matplotlib.pyplot as plt


def evaluate_cross_validation(clf, x, y, k):
    # create a k=fold cross validation iterator
    cv = KFold(k, shuffle=True, random_state=0)
    # score method of the estimator (accuracy)
    score = cross_val_score(clf, x, y, cv=cv)
    print(score)
    print('mean score: {0:.3f} (+/-{1:.3f})'.format(np.mean(score), sem(score)))

def train_and_evaluate(clf, x_train, x_test, y_train, y_test):
    clf.fit(x_train, y_train)
    print('Accuracy on training set:')
    print(clf.score(x_train, y_train))
    print('Accuracy on testing set:')
    print(clf.score(x_test, y_test))
    y_pred = clf.predict(x_test)

    print('classification report:')
    print(classification_report(y_test, y_pred))
    print('confusion matrix:')
    print(confusion_matrix(x_test, y_pred))

def create_target(num_sample,segments):
    # create a new y array of target size initialized with zeros
    y = np.zeros(num_sample)
    # put 1 in the specified segments
    for (start, end) in segments:
        y[start:end + 1] = 1
    return y


def print_faces(images, target, top_n):
    # set up figure size in inches
    fig = plt.figure(figsize=(12, 12))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(top_n):
        # we will print images in matrix 20x20
        p = fig.add_subplot(20, 20, i + 1, xticks=[], yticks=[])
        p.imshow(images[i], cmap=plt.cm.bone)
        # label the image with target value
        p.text(0, 14, str(target[i]))
        p.text(0, 60, str(i))


def main():
    faces = fetch_olivetti_faces()
    print(faces.DESCR)
    print(faces.keys())
    print(faces.images.shape)
    print(faces.data.shape)
    print(faces.target.shape)
    print(np.max(faces.data))
    print(np.min(faces.data))
    print(np.mean(faces.data))

    print_faces(faces.images, faces.target, 400)

    svc_1 = SVC(kernel='linear')
    # C-Support Vector Classification, based on libsvm
    # The multiclass support is handled according to a one-vs-one scheme.
    # kernel: 'linear', 'poly', 'rbf', 'sigmoid'
    print (svc_1)

    X_train, X_test, y_train, y_test = train_test_split(
            faces.data, faces.target, test_size=0.25, random_state=0)
    evaluate_cross_validation(svc_1, X_train, y_train, 5)
    train_and_evaluate(svc_1, X_train, X_test, y_train, y_test)

    # the index ranges of images of people with glasses
    glasses = [
        (10, 19), (30, 32), (37, 38), (50, 59), (63, 64),
        (69, 69), (120, 121), (124, 129), (130, 139), (160, 161),
        (164, 169), (180, 182), (185, 185), (189, 189), (190, 192),
        (194, 194), (196, 199), (260, 269), (270, 279), (300, 309),
        (330, 339), (358, 359), (360, 369)
    ]
    num_samples = faces.target.shape[0]
    target_glasses = create_target(num_samples, glasses)

    svc_2 = SVC(kernel='linear')
    X_train, X_test, y_train, y_test = train_test_split(
            faces.data, target_glasses, test_size=0.25, random_state=0)
    evaluate_cross_validation(svc_2, X_train, y_train, 5)
    train_and_evaluate(svc_2, X_train, X_test, y_train, y_test)

    X_test = faces.data[30:40]
    y_test = target_glasses[30:40]
    print (y_test.shape[0])
    select = np.ones(target_glasses.shape[0])
    select[30:40] = 0
    X_train = faces.data[select == 1]
    y_train = target_glasses[select == 1]
    print (y_train.shape[0])

    svc_3 = SVC(kernel='linear')
    train_and_evaluate(svc_3, X_train, X_test, y_train, y_test)
    y_pred = svc_3.predict(X_test)
    eval_faces = [np.reshape(a, (64, 64)) for a in X_test]
    print_faces(eval_faces, y_pred, 10)


if __name__ == "__main__":
    t_start = time.time()
    main()
    t_end = time.time()
    t_all = t_end - t_start
    print('whole time: {:.2f} min'.format(t_all / 60.))
