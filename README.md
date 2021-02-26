
# Classification Metrics


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
#your work here
```

#### Train-test split (`random_state` = 42) and standard scale all features

  - Why do we standardize *after* the train test split, and not before?

  - Why do we scale the training data separately from the testing data?


```python
#your work here
```

#### Create a logistic regression model with the first three features of the training data (with no regularization)


```python
#your work here
```

#### Get predictions for this 3-feature model for the training data

- Assign them to `train_preds_3`


```python
#your work here
```

#### Get predictions for this 3-feature model for the testing data

- Assign them to `test_preds_3`


```python
#your work here
```

#### Generate two confusion matrices, one each for the training predictions and testing predictions


```python
#your work here
```

#### Calculate the accuracy, recall, and precision for the training predictions

#### Calculate the accuracy, recall, and precision for the testing predictions


```python
#your work here
```

#### Is the model over- or under-fitting?  How can you tell?

#### Is bias or variance more of a problem with this model?


```python
#your work here
```

#### Run models with the first 10 variables, then another model with all the variables
  - Generate confusion matrices and calculate accuracy, precision and recall as you did above
  - **BONUS**: use functions to do so!
  
#### How is the problem you diagnosed in the 3-variable model altered in the 10-variable and 20-variable models?

#### What new problems crop up?


```python
#your work here
```
