#Part 1- Linear algebra and optimization
# In thís problem you will implement the gradient method to minimize a least square loss function for the regression problem. 
#First load the training data X and response variable y from X.txt and y.txt. 
#The least square loss function in the matrix form is given by:
# f(beta) = 1/2 * || y- X*bteta||_2^2. 
#You notice that there is no  intercept in this model. To make the implementation simple, we will skip the intercept here.

# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#load in the data
X = pd.read_csv(r'C:\Users\jacobsolomon\Downloads\X.txt', delimiter = ' ', header = None)
y = pd.read_csv(r'C:\Users\jacobsolomon\Downloads\y.txt', delimiter = ' ', header = None)

#create beta_0s
beta = np.zeros((50,1))

#1. Implement a function named loss(y,X,beta) which takes in 3 input arguments. The function will output the value of the
#least square loss function and its derivative in terms of beta. 

def loss(y,X,beta):
    value=.5 * np.linalg.norm(y - np.dot(X,beta))**2
    grad = np.dot(X.transpose(),np.dot(X,beta)) - np.dot(X.transpose(),y)
    return value,grad

beta_cur = np.zeros((50,1))
alpha=1e-4
maxiter=1000
f_cur,dev_cur=loss(y,X,beta_cur)
val=np.zeros((maxiter,1))

#2. Implement the gradient descent method to minimize this loss function. Use the starting point beta_0 as zeros, step-size
#alpha =1e-4, number iteration is 1000. Ouput the final estimate of beta and the corresponding objective function value.

for iter in range(maxiter):
    #calculate new estimate
    val[iter]=f_cur
    beta_new= beta_cur - alpha*dev_cur
    f_new, dev_new=loss(y,X,beta_new)
    if (abs(f_new-f_cur) < 1e-4):
        break
    beta_cur=beta_new
    f_cur=f_new
    dev_cur=dev_new

print(f_new)
print(beta_new)
plt.plot(val)
plt.show()

#Part 2 - Dimension reduction and classifiction
# For this assignment, we will use another digit data set, since it only consists of numerical features. Your job is to 
#apply the dimension reduction technique we learned, and combined with the classification methods you learned from DS861, 
#to build a classifier. 

import itertools
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding



X_digits, y_digits = load_digits(return_X_y = True)

X_digits = StandardScaler().fit_transform(X_digits)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, random_state = 862)

# create a pipeline for KNN
pipe = Pipeline([('clf', knn())])
search_space = [{'clf__n_neighbors': np.arange(2,4)}]

#define KPCA hyperparams
gamma_param = [0.01, 0.1, 1.0]
n_components_param = np.linspace(10,40,4, dtype = int)
combos = list(itertools.product(n_components_param,gamma_param))

#storing values for the models
kpca_training_score = np.zeros(len(combos))
best_classifier_params = []

#create index variable
p = 0

#loop through the KPCA hyperparams and fit to the classifier hyperparams pipeline
for c in combos:
    kpca = KernelPCA(n_components = c[0], kernel="rbf", gamma=c[1], fit_inverse_transform = True)
    kpca_transform = kpca.fit_transform(X_train)
    gs = GridSearchCV(pipe, search_space)
    gs.fit(kpca_transform, y_train)
    kpca_training_score[p] = gs.best_score_
    best_classifier_params.append(gs.best_params_['clf__n_neighbors'])
    p+=1

#define best hyperparams based on training score
best_params = np.argmax(kpca_training_score)
best_n_components =combos[best_params][0]
best_gamma = combos[best_params][1]

#fit and transform the training data with best KPCA parameters 
kpca_transform = KernelPCA(n_components = best_n_components, kernel="rbf", gamma=best_gamma, fit_inverse_transform = True)
kpca_train = kpca_transform.fit_transform(X_train)
#fit and transform the testing data with best KPCA parameters
kpca_test = kpca_transform.fit_transform(X_test)

#refit the KNN model with best hyperparams
train_knn_kpca = knn(n_neighbors = best_classifier_params[best_params]).fit(kpca_train,y_train)

#test score
test_score_knn_KPCA = train_knn_kpca.score(kpca_test,y_test)
train_score_knn_KPCA = train_knn_kpca.score(kpca_train,y_train)

########################################
### KPCA Part 2, Logistic Regression ###
########################################

# create a pipeline for Logistic Regression
pipe = Pipeline([('clf', LogisticRegression())])

#define LR hyperparams
search_space =     [{
                    'clf__penalty': ['l2'],
                    'clf__C': np.logspace(-10, 10, 4),
                    'clf__tol':  np.logspace(-10, 10, 4),
                    'clf__multi_class' : ['ovr'],
                    'clf__solver': ['liblinear']}]


#define KPCA hyperparams
gamma_param = [0.01, 0.1, 1.0]
n_components_param = np.linspace(10,40,4, dtype = int)
combos = list(itertools.product(n_components_param,gamma_param))

#storing values for the models
kpca_training_score = np.zeros(len(combos))
best_classifier_params = []

#create index variable
p = 0

X_digits, y_digits = load_digits(return_X_y = True)

X_digits = StandardScaler().fit_transform(X_digits)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, random_state = 862)
#loop through the KPCA hyperparams and fit to the classifier hyperparams pipeline
for c in combos:
    kpca = KernelPCA(n_components = c[0], kernel="rbf", gamma=c[1], fit_inverse_transform = True)
    kpca_transform = kpca.fit_transform(X_train)
    gs = GridSearchCV(pipe, search_space)
    gs.fit(kpca_transform, y_train)
    kpca_training_score[p] = gs.best_score_
    best_classifier_params.append([gs.best_params_['clf__C'],gs.best_params_['clf__tol']])
    p+=1


#define best hyperparams based on training score
best_params = np.argmax(kpca_training_score)
best_n_components =combos[best_params][0]
best_gamma = combos[best_params][1]
best_C = best_classifier_params[best_params][0]
best_tol = best_classifier_params[best_params][1]

#fit and transform the training data with best KPCA parameters 
kpca_transform = KernelPCA(n_components = best_n_components, kernel="rbf", gamma=best_gamma, fit_inverse_transform = True)
kpca_train = kpca_transform.fit_transform(X_train)

#fit and transform the testing data with best KPCA parameters
kpca_test = kpca_transform.fit_transform(X_test)

#refit the KNN model with best hyperparams
train_lr_kpca = LogisticRegression(penalty = 'l2', multi_class = 'ovr', solver = 'liblinear', C = best_C, tol = best_tol ).fit(kpca_train,y_train)

#test score
test_score_lr_kpca = train_lr_kpca.score(kpca_test,y_test)
train_score_lr_kpca = train_lr_kpca.score(kpca_train,y_train)

###########
### LLE ###
###########

# create a pipeline for KNN
pipe = Pipeline([('clf', knn())])
search_space = [{'clf__n_neighbors': np.arange(2,4)}
               ]

#define LLE hyperparams
n_components_param = np.linspace(10,40,4, dtype = int)
n_neighbors = [2,5,10]
combos = list(itertools.product(n_components_param,n_neighbors))

#storing values for the models
lle_training_score = np.zeros(len(combos))
best_classifier_params = []

#create index variable
p = 0

#loop through the LLE hyperparams and fit to the classifier hyperparams pipeline
for c in combos:
    X_reduced_lle = LocallyLinearEmbedding(n_components = c[0], n_neighbors=c[1], random_state=862, n_jobs = -1)
    lle_transform = X_reduced_lle.fit_transform(X_train)
    gs = GridSearchCV(pipe, search_space)
    gs.fit(lle_transform, y_train)
    lle_training_score[p] = gs.best_score_
    best_classifier_params.append(gs.best_params_['clf__n_neighbors'])
    p+=1

#define best hyperparams based on training score
best_params = np.argmax(lle_training_score)
best_n_components =combos[best_params][0]
best_nn = combos[best_params][1]

#fit and transform the training data with best LLE parameters 
lle_transform = LocallyLinearEmbedding(n_components = best_n_components, n_neighbors=best_nn, random_state=862, n_jobs = -1)
lle_train = lle_transform.fit_transform(X_train)

#fit and transform the testing data with best LLE parameters
lle_test = lle_transform.fit_transform(X_test)

#refit the KNN model with best hyperparams
train_knn_lle = knn(n_neighbors = best_classifier_params[best_params]).fit(lle_train,y_train)

#test and training score
test_score_knn_lle = train_knn_lle.score(lle_test,y_test)
train_score_knn_lle = train_knn_lle.score(lle_train,y_train)

#######################################
### LLE Part 2, Logistic Regression ###
#######################################

# create a pipeline for Logistic Regression
pipe = Pipeline([('clf', LogisticRegression())])

#define LR hyperparams
search_space =     [{
                    'clf__penalty': ['l2'],
                    'clf__C': np.logspace(-10, 10, 4),
                    'clf__tol':  np.logspace(-10, 10, 4),
                    'clf__multi_class' : ['ovr'],
                    'clf__solver': ['liblinear']}]


#define LLE hyperparams
n_components_param = np.linspace(10,40,4, dtype = int)
n_neighbors = [2,5,10]
combos = list(itertools.product(n_components_param,n_neighbors))

#storing values for the models
lle_training_score = np.zeros(len(combos))
best_classifier_params = []

#create index variable
p = 0

#loop through the LLE hyperparams and fit to the classifier hyperparams pipeline
for c in combos:
    X_reduced_lle = LocallyLinearEmbedding(n_components = c[0], n_neighbors=c[1], random_state=862, n_jobs = -1)
    lle_transform = X_reduced_lle.fit_transform(X_train)
    gs = GridSearchCV(pipe, search_space)
    gs.fit(lle_transform, y_train)
    lle_training_score[p] = gs.best_score_
    best_classifier_params.append([gs.best_params_['clf__C'],gs.best_params_['clf__tol']])
    p+=1


#define best hyperparams based on training score
best_params = np.argmax(lle_training_score)
best_n_components =combos[best_params][0]
best_nn = combos[best_params][1]
best_C = best_classifier_params[best_params][0]
best_tol = best_classifier_params[best_params][1]

#fit and transform the training data with best LLE parameters 
lle_transform = LocallyLinearEmbedding(n_components = best_n_components, n_neighbors=best_nn, random_state=862, n_jobs = -1)
lle_train = lle_transform.fit_transform(X_train)

#fit and transform the testing data with best LLE parameters
lle_test = lle_transform.fit_transform(X_test)

#refit the LR model with best hyperparams
train_lr_lle = LogisticRegression(penalty = 'l2', multi_class = 'ovr', solver = 'liblinear', C = best_C, tol = best_tol ).fit(lle_train,y_train)

#test score
test_score_lr_lle = train_lr_lle.score(lle_test,y_test)
train_score_lr_lle = train_lr_lle.score(lle_train,y_train)


##############
### Isomap ###
##############

isomap = Isomap(n_components = 2, n_neighbors = 10, n_jobs = -1)
X_reduced_isomap = isomap.fit_transform(X)

# create a pipeline for KNN
pipe = Pipeline([('clf', knn())])
search_space = [{'clf__n_neighbors': np.arange(2,4)}
               ]

#define isomap hyperparams
n_components_param = np.linspace(10,40,4, dtype = int)
n_neighbors = [2,5,10]
combos = list(itertools.product(n_components_param,n_neighbors))

#storing values for the models
isomap_training_score = np.zeros(len(combos))
best_classifier_params = []

#create index variable
p = 0

#loop through the isomap hyperparams and fit to the classifier hyperparams pipeline
for c in combos:
    X_reduced_isomap = Isomap(n_components = c[0], n_neighbors=c[1],  n_jobs = -1)
    isomap_transform = X_reduced_isomap.fit_transform(X_train)
    gs = GridSearchCV(pipe, search_space)
    gs.fit(isomap_transform, y_train)
    isomap_training_score[p] = gs.best_score_
    best_classifier_params.append(gs.best_params_['clf__n_neighbors'])
    p+=1

#define best hyperparams based on training score
best_params = np.argmax(isomap_training_score)
best_n_components =combos[best_params][0]
best_nn = combos[best_params][1]

#fit and transform the training data with best isomap parameters 
isomap_transform = Isomap(n_components = best_n_components, n_neighbors=best_nn, n_jobs = -1)
isomap_train = isomap_transform.fit_transform(X_train)

#fit and transform the testing data with best isomap parameters
isomap_test = isomap_transform.fit_transform(X_test)

#refit the KNN model with best hyperparams
train_knn_isomap = knn(n_neighbors = best_classifier_params[best_params]).fit(isomap_train,y_train)

#test score
test_score_knn_isomap = train_knn_isomap.score(isomap_test,y_test)
train_score_knn_isomap = train_knn_isomap.score(isomap_train,y_train)

##########################################
### Isomap Part 2, Logistic Regression ###
##########################################

# create a pipeline for Logistic Regression
pipe = Pipeline([('clf', LogisticRegression())])

#define LR hyperparams
search_space =     [{
                    'clf__penalty': ['l2'],
                    'clf__C': np.logspace(-10, 10, 4),
                    'clf__tol':  np.logspace(-10, 10, 4),
                    'clf__multi_class' : ['ovr'],
                    'clf__solver': ['liblinear']}]


#define isomap hyperparams
n_components_param = np.linspace(10,40,4, dtype = int)
n_neighbors = [2,5,10]
combos = list(itertools.product(n_components_param,n_neighbors))

#storing values for the models
lle_training_score = np.zeros(len(combos))
best_classifier_params = []

#create index variable
p = 0

#loop through the isomap hyperparams and fit to the classifier hyperparams pipeline
for c in combos:
    X_reduced_isomap = Isomap(n_components = c[0], n_neighbors=c[1], n_jobs = -1)
    isomap_transform = X_reduced_isomap.fit_transform(X_train)
    gs = GridSearchCV(pipe, search_space)
    gs.fit(isomap_transform, y_train)
    isomap_training_score[p] = gs.best_score_
    best_classifier_params.append([gs.best_params_['clf__C'],gs.best_params_['clf__tol']])
    p+=1


#define best hyperparams based on training score
best_params = np.argmax(isomap_training_score)
best_n_components =combos[best_params][0]
best_nn = combos[best_params][1]
best_C = best_classifier_params[best_params][0]
best_tol = best_classifier_params[best_params][1]

#fit the transform training data with best isomap parameters 
isomap_transform = LocallyLinearEmbedding(n_components = best_n_components, n_neighbors=best_nn, n_jobs = -1)
isomap_train = isomap_transform.fit_transform(X_train)
#fit and transform the testing data with best isomap parameters
isomap_test = isomap_transform.fit_transform(X_test)

#refit the LR model with best hyperparams
train_lr_isomap = LogisticRegression(penalty = 'l2', multi_class = 'ovr', solver = 'liblinear', C = best_C, tol = best_tol ).fit(isomap_train,y_train)

#test and train score for isomap LR
isomap_lr_test_score = train_lr_isomap.score(isomap_test,y_test)
isomap_lr_train_score = train_lr_isomap.score(isomap_train,y_train)

compare_df = pd.DataFrame(data = {'Test_Score': [test_score_knn_KPCA,test_score_lr_kpca,test_score_knn_lle,test_score_lr_lle,test_score_knn_isomap,isomap_lr_test_score],
                                  'Train_Score':  [train_score_knn_KPCA,train_score_lr_kpca,train_score_knn_lle,train_score_lr_lle,train_score_knn_isomap,isomap_lr_train_score]},
                          index = ['KPCA_KNN','KPCA_LR','LLE_KNN','LLE_LR','ISOMAP_KNN','ISOMAP_LR'])

best_training_score = compare_df[compare_df.Train_Score ==compare_df.Train_Score.max()]
best_test_score = compare_df[compare_df.Train_Score ==compare_df.Train_Score.max()]

# What is the best combination according to your accuracy score on the test set?

# Answer: KPCA with linear regression

print(best_training_score)
print(best_test_score)

# Now using the original data set and the two classifers you chose, run the procedure again, but this time without any dimension reduction.
# Make sure you tune your classifiers. Which result is better? Using the original data set or the reduced data set?

# create a pipeline for KNN without dimmension reduction
pipe = Pipeline([('clf', knn())])
search_space = [{'clf__n_neighbors': np.arange(2,4)}]
gs = GridSearchCV(pipe, search_space)
gs.fit(X_train,y_train)
knn_train = gs.best_score_ #97.5% accuracy for training
knn_test = gs.score(X_test,y_test) # 98% accuracy for test

# create a pipeline for logistic regression without dimmension reduction
pipe = Pipeline([('clf', LogisticRegression())])

search_space =     [{
                    'clf__penalty': ['l2'],
                    'clf__C': np.logspace(-10, 10, 4),
                    'clf__tol':  np.logspace(-10, 10, 4),
                    'clf__multi_class' : ['ovr'],
                    'clf__solver': ['liblinear']}]
gs = GridSearchCV(pipe, search_space)
gs.fit(X_train,y_train)
lr_train = gs.best_score_ #94.7% accuracy for training
lr_test = gs.score(X_test,y_test) # 92.6% accuracy for test

# Answer: KNN with all of the features did the best out of all of my trials for the test data. 
# All of the test scores I have calculated for the dimmension reduction techniques are very low where as their training is quite high. It could be that the model has overfit to the data but I doubt
# it would make such awful test scores as I am seeing in my dimmension reduction dataframe results. It is most likely a bug in my code sadly. However the best training data score I have for KPCA using linear regression
# is almost 100% accurate which is quite incredible. My non dimmension reducing results were nowhere near that for logistic regression. 

# The KNN results for no dimmension reduction were also quite high for both training and testing. 97.5 for training and 98% for testing which is quite surprising to see testing is greater than training.
#Overall this was a great excersize in for loops and the procedures of GridsearchCV and Pipeline. However I am sure there are easier ways to execute the same procedures as I have with the use of classes.
#I tried to use your links however I just could not get it to work properly. Thank you for sharing them though!
