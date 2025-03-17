
# **Convert Class**

Convert is a preprocessing utility class designed to automatically identify numerical, categorical, and ordinal columns in a DataFrame, convert appropriate object columns to numeric types, and prepare the data for machine learning models.

## **Initialization**

```python
Convert(df, target)
```

**Parameters:**

- `df` (pandas.DataFrame): The input dataset.
- `target` (str): The name of the target column (this column will be excluded from feature processing).

## **Method: apply()**

```python
apply(ncol=None, ocol=None, ordinal_cols=None, ord_threshold=None, ordname=(), drop_cols=(), typeval=80, convrate=80)
```

**Parameters:**

- `ncol` (list, default None): List of numerical columns. Auto-inferred if not provided.
- `ocol` (list, default None): List of categorical columns. Auto-inferred if not provided.
- `ordinal_cols` (dict, default {}): Dictionary mapping ordinal columns to their ordered categories.
- `ord_threshold` (int, default None): Treat object columns with unique values ≤ threshold as ordinal.
- `ordname` (tuple/list/str, default ()): Manually specified ordinal column names.
- `drop_cols` (list/tuple, default ()): Columns to drop from processing.
- `typeval` (int, default 80): % threshold to convert object column to numeric if ≥ this % numeric.
- `convrate` (int, default 80): % threshold to decide int/float conversion.

**Returns:**

- `X` (DataFrame): Processed feature DataFrame.
- `y` (Series): Target column.
- `ncol` (list): Final list of numerical columns.
- `ocol` (list): Final list of categorical columns.
- `ordinal_cols` (dict): Final dictionary of ordinal columns.

---

# **Brain Class**

Brain is an end-to-end AutoML pipeline class that handles data preprocessing, encoding, imputation, scaling, PCA, duplicate handling, hyperparameter tuning, and model training for both classification and regression tasks.

## **Initialization**

```python
Brain(df, target, model_name, task='classification', scale=False, grid_search=None,
      test_size=0.2, cv=5, voteclass=[], numerical_cols=[], categorical_cols=[],
      ordinal_encoding=True, ordinal_cols={}, ynan=True, categorical_encoding='onehot',
      numerical_impute_strategy='mean', showx=False, summary=False, miss=False,
      categorical_impute_strategy='most_frequent', ordinal_impute_strategy='most_frequent',
      preprocessed_out=False, yencode='encode', categorical_fill_value=None, ordinal_fill_value=None,
      pca=False, pca_comp=0.9, objvalue=True, drop_duplicates=False, xtype=False, voting='soft', pa=0,
      dropc=False, column_value={}, ord_threshold=None, ordname=(), drop_cols=(), typeval=80, convrate=80)
```

**Key Parameters:**

- `df` (DataFrame): Input dataset.
- `target` (str): Target column name.
- `model_name` (str): Model name (e.g., 'RFC', 'XGBC', 'Linear', etc.).
- `task` (str, default 'classification'): Task type, 'classification' or 'regression'.
- `scale` (bool, default False): Apply scaling to numerical features.
- `grid_search` (str, default None): Hyperparameter tuning method, 'cv', 'rand', or None.
- `test_size` (float, default 0.2): Proportion of dataset used for test set.
- `cv` (int, default 5): Number of cross-validation folds.
- `voteclass` (list, default []): Classifiers used in voting ensemble.
- `numerical_cols` (list, default []): List of numerical columns. Auto-detected if empty.
- `categorical_cols` (list, default []): List of categorical columns. Auto-detected if empty.
- `ordinal_encoding` (bool, default True): Apply ordinal encoding if True.
- `ordinal_cols` (dict, default {}): Dictionary mapping ordinal columns to ordered categories.
- `ynan` (bool, default True): Drop rows where target column has NaN values.
- `categorical_encoding` (str, default 'onehot'): Encoding method for categorical columns ('onehot' or 'label').
- `numerical_impute_strategy` (str, default 'mean'): Strategy to impute numerical columns.
- `showx` (bool, default False): Display processed feature set if True.
- `summary` (bool, default False): Show summary of feature sets if True.
- `miss` (bool, default False): Show missing value report if True.
- `categorical_impute_strategy` (str, default 'most_frequent'): Strategy to impute categorical columns.
- `ordinal_impute_strategy` (str, default 'most_frequent'): Strategy to impute ordinal columns.
- `preprocessed_out` (bool, default False): Return preprocessed data if True.
- `yencode` (str, default 'encode'): Target encoding method ('encode' or 'bin').
- `categorical_fill_value` (any, default None): Fill value for categorical imputation.
- `ordinal_fill_value` (any, default None): Fill value for ordinal imputation.
- `pca` (bool, default False): Apply PCA dimensionality reduction if True.
- `pca_comp` (float, default 0.9): PCA components or variance ratio to retain.
- `objvalue` (bool, default True): Display value counts of object columns if True.
- `drop_duplicates` (bool or str, default False): Handle duplicate rows. Options: False, True, or 'all'.
- `xtype` (bool, default False): Show column data types if True.
- `voting` (str, default 'soft'): Voting strategy in ensemble classifiers ('soft' or 'hard').
- `pa` (int, default 0): Index of parameter plotted during hyperparameter search.
- `dropc` (bool, default False): Enable row dropping based on column values.
- `column_value` (dict, default {}): Columns and values to filter out rows.
- `ord_threshold` (int, default None): Columns with unique values <= threshold treated as ordinal.
- `ordname` (tuple/list/str, default ()): Manually specified ordinal column names.
- `drop_cols` (list/tuple, default ()): Columns to exclude from processing.
- `typeval` (int, default 80): % threshold to convert object column to numeric if ≥ this % numeric.
- `convrate` (int, default 80): % threshold to decide int/float conversion.


**Conversion-related Parameters:**

- `ord_threshold`, `ordname`, `drop_cols`, `typeval`, `convrate`: Passed to internal convert() function.

## **Methods**

### **convert()**

Wrapper of Convert.apply() functionality, used internally to process features and extract column types if not provided explicitly.

### **preprocess_data()**

Prepares data with imputation, encoding, scaling, and handles ordinal and categorical features.

### **build()**

Runs the full pipeline: converts, preprocesses, splits, tunes, and trains the model. Outputs evaluation metrics and optionally returns the trained model or preprocessed data.

## **Usage Example**

```python
from sklearn.datasets import load_iris
from brainsbuildcode import Convert, Brain

# Load dataset
data = load_iris(as_frame=True)
df = data.frame
df['target'] = data.target
target = 'target'

# Preprocess
converter = Convert(df, target)
X, y, ncol, ocol, ordinal_cols = converter.apply()

# Train
brain = Brain(df=df, target=target, model_name='RFC', task='classification',
              numerical_cols=ncol, categorical_cols=ocol, ordinal_cols=ordinal_cols)
brain.build()
```

