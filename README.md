# Brains Build Code - Automated Machine Learning Pipeline

Brains Build Code is an automated machine learning pipeline that performs data preprocessing, feature engineering, model selection, hyperparameter tuning, and evaluation.

## Installation

pip install brainsbuildcode


# Usage
```python
from brainsbuildcode import Brain

from sklearn.datasets import load_breast_cancer

import pandas as pd

data = load_breast_cancer(as_frame=True)

df = data.frame

# Train and build the model

best_model = Brain(df, target='target', model_name='RFC', grid_search=None)

best_model.build()

# alternative way to build the model

best_model = Brain(df, target='target', model_name='RFC', grid_search='cv').build()

```
# Install directly from GitHub:

pip install git+https://github.com/achelousace/brainsbuildcode.git
