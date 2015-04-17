
__author__ = 'kevintandean'
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier



def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print '%r %2.2f sec' % \
              (method.__name__, te-ts)
        return result

    return timed

def load_and_clean(path):
    df = pd.read_csv(path, sep='\t')
    descriptors = df.iloc[:, 15:150]
    logbb = df['LOG BB']

    def binarize(x):
        if x == 'BBB+':
            return 1
        elif x == 'BBB-':
            return 0
        elif x > 0:
            return 1
        elif x <= 0:
            return 0


    logbb = logbb.apply(binarize)
    positive=0
    negative=0
    for i in logbb:
        if i == 1:
            positive +=1
        elif i == 0:
            negative +=1

    descriptors = descriptors.join(logbb)

    rows_with_error  = descriptors.apply(
           lambda row : any([ e == '#NAME?' or np.isfinite(float(e))==False for e in row ]), axis=1)

    descriptors = descriptors[~rows_with_error]



    descriptors = descriptors.applymap(lambda x: float(x))

    return descriptors

descriptors = load_and_clean('all data full descriptors.txt')
print descriptors.head()


def split(data, size):
    grouped = data.groupby('LOG BB')
    bbb_neg = grouped.get_group(0.0)
    bbb_pos = grouped.get_group(1.0)


    x_pos = bbb_pos.iloc[:929,0:135].values
    y_pos = bbb_pos.iloc[:929,135:136].values
    x_pos_train, x_pos_test, y_pos_train, y_pos_test = train_test_split(x_pos, y_pos, test_size=size, random_state=100)

    x_neg = bbb_neg.iloc[:,0:135].values
    y_neg = bbb_neg.iloc[:,135:136].values
    x_neg_train, x_neg_test, y_neg_train, y_neg_test = train_test_split(x_neg, y_neg, test_size=size, random_state=100)

    x_train = np.append(x_pos_train, x_neg_train, axis = 0)
    y_train = np.append(y_pos_train, y_neg_train, axis = 0)
    x_test = np.append(x_pos_test, x_neg_test, axis = 0)
    y_test = np.append(y_pos_test, y_neg_test, axis = 0)


    return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = split(descriptors, 0.3)

from sklearn.feature_selection import chi2, f_classif
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import RFE

estimator = svm.SVC(kernel='linear')
selector = RFE(estimator,50)
selector = selector.fit(x_train,y_train)
x_train_sel = selector.transform(x_train)
print x_train_sel.shape

# import pickle
# s = pickle.dumps(selector)

from sklearn.externals import joblib
joblib.dump(selector, 'filename.pkl')



# forest = RandomForestClassifier()
# forest.fit(x_train,y_train)
# print forest.score(x_test,y_test)
#
# clf = svm.SVC()
# clf.fit(x_train,y_train)
# print clf.score(x_test,y_test)


#
# for i in [0.1,0.2,0.3,0.4]:
#     x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=i, random_state=42)
#     # print x_train
#     # print y_train
#     # print x_test
#     # print y_test
#
#
#     clf.fit(x_train,y_train)
#     forest.fit(x_train,y_train)
#     print i
#     print clf.score(x_test,y_test)
#     print forest.score(x_test,y_test)