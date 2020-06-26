# Do you even compare the metrics of your models bro


```python

#run as-is

import pandas as pd

from sklearn.datasets import make_classification

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

data = make_classification(n_samples=10000, random_state=666, n_informative=6)

X = pd.DataFrame(data[0])
y = data[1]

data = X.copy()
data['target'] = y
```

#### How many features in `data`?  How many classes?  Is there a class imbalance?


```python



'''
20

2

nope!
'''
```

#### Train-test split (`random_state` = 666) and standard scale all features

  - Why do we standardize *after* the train test split, and not before?

  - Why do we scale the training data separately from the testing data?


```python

X = data.iloc[:, :20]
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=666)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
```

#### Create a logistic regression model with the first three features of the training data (with no regularization)


```python

X_train_3 = (
    pd.DataFrame(X_train)
    .iloc[:,:3]
)

log_reg = LogisticRegression(penalty='none') 

log_reg.fit(X_train_3, y_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='auto', n_jobs=None, penalty='none',
                       random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                       warm_start=False)



#### Get predictions for this 3-feature model for the training data

- Assign them to `train_preds_3`


```python

train_preds_3 = log_reg.predict(X_train_3)
```

#### Get predictions for this 3-feature model for the testing data

- Assign them to `test_preds_3`


```python

X_test_3 = (
    pd.DataFrame(X_test)
    .iloc[:,:3]

)

test_preds_3 = log_reg.predict(X_test_3)
```

#### Generate two confusion matrices for the training predictions and testing predictions


```python

train_cm = confusion_matrix(y_train, train_preds_3)
test_cm = confusion_matrix(y_test, test_preds_3)

print(train_cm)
print(test_cm)
```

#### Calculate the accuracy, recall, and precision for the training predictions

#### Calculate the accuracy, recall, and precision for the testing predictions


```python

tn, fp, fn, tp = train_cm.ravel()

print(f'''
training 
accuracy: {(tn+tp)/len(X_train)}
precision: {(tp)/(tp+fp)} 
recall: {tp/(tp+fn)}
'''
)
print()

tn, fp, fn, tp = test_cm.ravel()

print(f'''
test 
accuracy: {(tn+tp)/len(X_test)}
precision: {(tp)/(tp+fp)} 
recall: {tp/(tp+fn)}
'''
)
```

    
    training 
    accuracy: 0.734875
    precision: 0.7315502724120851 
    recall: 0.7401653720871962
    
    
    
    test 
    accuracy: 0.7305
    precision: 0.7272727272727273 
    recall: 0.7410358565737052
    


#### Is the model over- or under-fitting?  How can you tell?

#### Is bias or variance more of a problem with this model?


```python

'''
Underfitting, because the train and test error are fairly close, but they're both low and can be improved

'''
```

#### Run models with the first 10 variables, then another model with all the varibles
  - Generate confusion matrices and calculate accuracy, precision and recall as you did above
  - **BONUS**: use functions to do so!
  
#### How is the problem you diagnosed in the 3-variable model altered in the 10-variable and 20-variable models?

#### What new problems crop up?


```python

def make_cms(xtrain, xtest, ytrain, ytest, model_obj):
    '''
    returns train and test confusion matrices for a given model
    '''
    
    model_obj.fit(xtrain, ytrain)
    
    preds_train = model_obj.predict(xtrain)
    
    preds_test = model_obj.predict(xtest)    
    
    cm_train = confusion_matrix(ytrain, preds_train)
    
    cm_test = confusion_matrix(ytest, preds_test)
    
    return cm_train, cm_test

def cm_calcs(cm, label):
    '''
    calculate accuracy, precision and recall from a confusion matrix
    '''
    
    tn, fp, fn, tp = cm.ravel()

    print(f'''
    {label} 
    accuracy: {(tn+tp)/(tn+tp+fn+fp)}
    precision: {(tp)/(tp+fp)} 
    recall: {tp/(tp+fn)}
    '''
    )
    print()
    
    return
    
    
def model_calc(xtrain, xtest, ytrain, ytest, model_obj, train_label, test_label):
    cm_train, cm_test = make_cms(xtrain, xtest, ytrain, ytest, model_obj)
    
    cm_calcs(cm_train, train_label)
    cm_calcs(cm_test, test_label)
    
    return

#10-feature model
X_train_10 = pd.DataFrame(X_train).iloc[:,:10]
X_test_10 = pd.DataFrame(X_test).iloc[:,:10]

model_calc(X_train_10, X_test_10, y_train, y_test, 
           LogisticRegression(penalty='none'), 
          'training metrics for 10-variable model',
          'testing metrics for 10-variable model')

#20-feature model
model_calc(X_train, X_test, y_train, y_test,
           LogisticRegression(penalty='none'), 
          'training metrics for 20-variable model',
          'testing metrics for 20-variable model')

print('''
Underfitting is much less of a problem for the 10-variable and 20-variable models.

Accuracy, precision and recall all took huge jumps into the range above 90%.

The metrics for the 10-variable model are close together for the train and
test set of the 10-variable model, indicating relatively low bias and variance

The metrics for the 20-variable model are slighly further apart for the 
train and test set, indicating that overfitting is starting to creep
in slightly.

''')
```

    
        training metrics for 10-variable model 
        accuracy: 0.934375
        precision: 0.9257985257985258 
        recall: 0.9441242796291657
        
    
    
        testing metrics for 10-variable model 
        accuracy: 0.9265
        precision: 0.9213372664700098 
        recall: 0.9332669322709163
        
    
    
        training metrics for 20-variable model 
        accuracy: 0.934125
        precision: 0.925343811394892 
        recall: 0.9441242796291657
        
    
    
        testing metrics for 20-variable model 
        accuracy: 0.9255
        precision: 0.9195289499509323 
        recall: 0.9332669322709163
        
    
    
    Underfitting is much less of a problem for the 10-variable and 20-variable models.
    
    Accuracy, precision and recall all took huge jumps into the range above 90%.
    
    The metrics for the 10-variable model are close together for the train and
    test set of the 10-variable model, indicating relatively low bias and variance
    
    The metrics for the 20-variable model are slighly further apart for the 
    train and test set, indicating that overfitting is starting to creep
    in slightly.
    
    

