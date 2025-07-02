<h1> Optimization </h1>

--------------------------------------------------------------------------------------

## optuna_study

```python
class opt.optuna_study()
```

An encapsulated interface for creating, loading, configuring, and running  
hyperparameter optimization with Optuna, including support for dynamic range adjustment,  
scoring strategies, and exportable checkpoints.

This class simplifies integration of Optuna with machine learning models by allowing:  
- Direct instantiation of new or existing studies.  
- Flexible definition of hyperparameter search spaces.  
- Objective functions with default scoring or cross-validation scoring.  
- Multiple stopping criteria for early stopping or range expansion.  
- Automatic saving of checkpoints as `.pkl` files.

**Parameters:**

- `model`: `object`
    A scikit-learn-like estimator class (e.g., `RandomForestRegressor`) that will be optimized.

- `x_train`: `pd.DataFrame`
    The training feature dataset.

- `y_train`: `pd.DataFrame`
    The training target dataset.

**Default Parameters:**

- `study = 'new'`: `str` or `optuna.study.Study` or `optuna_study`
    Defines how to initialize the Optuna study. Can be:
    - `'new'`: create a new study;
    - `'str`: A string path to a `.pkl` file containing a saved study object;
    - `optuna.study.Study`: An existing Optuna `Study` object.

- `study_name = None`: `str`    
    Name for the study if creating a new one.

- `direction = None`: `str`    
    Direction of optimization: either `'minimize'` or `'maximize'` (used if creating a new study).

**Attributes:**

- `model`: `type`
    The machine learning estimator class to be tuned, e.g., `RandomForestRegressor`, `XGBRegressor`, or any scikit-learn-like model class.

- `x_train`: `pd.DataFrame`  
    The training feature dataset; a pandas DataFrame containing the input features for model training.

- `y_train`: `pd.DataFrame` or `pd.Series` 
     The training target dataset; a pandas DataFrame or Series containing the target variable(s).

- `study`; `optuna.study.Study`
    The `optuna.study.Study` object that manages and stores optimization trials.

- `study_name`: `str` or `None`
    Name identifier for the Optuna study. Used when creating a new study.

- `direction`: `str`, valid values: `'minimize'` or `'maximize'`  
    Defines whether the optimization goal is to minimize or maximize the objective function.

- `scorer`: `callable` or `dict`  
    The scoring function or a dictionary of scoring functions used to evaluate model performance during optimization.

- `_objective_mode`: `str`, valid values: `'default'` or `'cv'`  
    Indicates the mode of the objective function evaluation: `'default'` for a single scorer on the training set or `'cv'` for cross-validation scoring.

- `_fobj_mult`: `float` or `int`  
    A multiplier applied to the objective function score, typically used to scale the score for better numerical stability during optimization.

- `_objective_mode_default`: `dict`  
    Configuration dictionary for the `'default'` mode containing keys such as `'scorer'` and `'scorer_locals'` with corresponding values.

- `_objective_mode_cv`: `dict`  
    Configuration dictionary for the `'cv'` mode containing keys such as `'cv_scorers'`, `'cv_scorers_weight'`, `'cv'`, `'use_cv_train'`, and `'cv_train_weight'`.

- `_hparams`: `dict`  
    Dictionary holding the hyperparameter search space and fixed model arguments, divided by types: categorical (`'cat'`), integer (`'int'`), float (`'float'`), and fixed model parameters (`'model'`).


**Methods:**

- `describe_study()`: type, valid values 
    Prints a summary of the current study including its name, model, number of trials, best score, dataset shape, optimization direction, and whether cross-validation is used.

- `set_objective_conditions(...)`: type, valid values 
    Sets up the hyperparameter search space and the objective evaluation conditions, including scoring functions, mode (default or CV), and weights.

- `_create_objective()`: type, valid values 
    Builds the objective function that Optuna will optimize, based on the set conditions. It can be called after `set_objective_conditions(...)` is called to construct a function for other applications.

- `optimize(...)`: type, valid values 
    Runs the optimization process with flexible stopping criteria, supports checkpointing, and dynamic range adjustment of hyperparameters.

**Examples**
```python
from sklearn.ensemble import RandomForestRegressor
opt_study = optuna_study(
    model=RandomForestRegressor,
    x_train=X,
    y_train=y,
    direction='minimize'
)
```
After define study:
```python
opt_study.set_objective_conditions( ... )

opt_study.optimize( ... )
```

---

### method: `describe_study()`

Prints a summary description of the current Optuna study, including the study name, model type, number of trials, best score found, dataset size, optimization direction, scoring function used, and whether cross-validation is enabled.

---

### method: `set_objective_conditions()`

Configures the objective function to be used during optimization. Defines the hyperparameter search space and scoring strategy. Supports single scoring (`default` mode) and cross-validation scoring (`cv` mode) with multiple metrics and weighted aggregation.

The hyperparameters must be provided as dictionaries with the format:
```python
{parameter_name: [suggest_args, suggest_kwargs]}
```

where `suggest_args` are positional arguments for Optuna’s suggest methods (e.g., ranges) and `suggest_kwargs` are optional keyword arguments (e.g., steps or logarithmic search).

**Default Parameters:**

- `hparams_cat = None`: `dict`
    Categorical hyperparameters to optimize. Format: `{param_name: [choices_list, {}]}`.

- `hparams_int = None`: `dict`
    Integer hyperparameters to optimize.
    Format:

```python
    {param_name: [[low, high], {suggest_kwargs}]}
```

- `hparams_float = None`: `dict`
    Float hyperparameters to optimize.
    Format:
```python
    {param_name: [[low, high], {suggest_kwargs}]}
```

- `model_locals = None`: `dict`
    Fixed model parameters (not optimized).

- `mode = 'default'`: `str`
    Objective evaluation mode:  
        - `'default'`: Uses a single scorer function on the full training set.  
        - `'cv'`: Uses multiple scorers combined with cross-validation.


- `scorer = None`: `callable` or `str`
    Scorer function or predefined scorer name (e.g., `'rmse'`, `'r2'`) for `'default'` mode.

- `scorer_locals = {}`: `dict`
    Keyword arguments passed to the scorer function.

- `cv_scorers = None`: `str`, `callable`, or `list`
    One or multiple scorers to use during cross-validation. Each can be a string name, a callable function `(y_true, y_pred)`, or a sklearn metrics object.

- `cv_scorers_weight = None`: `float` or `list`
    Weights for each CV scorer when aggregating results.

- `cv = 5`: `int` 
    Number of folds for cross-validation.

- `use_cv_train = False`: `bool` 
    If `True`, includes training scores in the CV aggregation.

- `cv_train_weight = 0`: `float` 
    Weight of training scores in aggregation (range 0–1).

- `fobj_mult = 1000`: `float` 
    Multiplier applied to the objective value to scale the score.

**Examples**

```python
opt_study.set_objective_conditions(
    hparams_float={'alpha': [[0.001, 1], {'log': True}]},
    model_locals={'random_state': 42},
    scorer='r2',
    mode='default'
)

opt_study.set_objective_conditions(
    hparams_int={'max_depth': [[3, 10], {}]},
    cv_scorers=['r2', 'rmse'],
    cv_scorers_weight=[0.6, 0.4],
    mode='cv'
)
```
For SVR model:
```python
#hparams = { attribute_name : [ optuna_suggest_args, 
#                               optuna_suggest_kargs ] }
hparams_float = {
    'C': ( [1e-3, 1e3], {'step':0.001} ),
    'gamma': ( [1e-4, 1e2], {'step':0.0001} ),
    'epsilon': ( [0.001, 1.0], {'step':0.001} ) 
    }

opt_study = optuna_study(SVR, 
                     x_train, 
                     y_train,
                     study='new',
                     study_name=f'SVR_with_cv',
                     direction='maximize',
                     )

opt_study.set_objective_conditions(
    hparams_float = hparams_float,
    mode='cv',
    cv_scorers='r2',
    )
```
For RF model:
```python
#hparams = { attribute_name : [ optuna_suggest_args, 
#                               optuna_suggest_kargs ] }
hparams_cat = {'criterion': ['squared_error', 'absolute_error'] }
hparams_int = {
    'n_estimators': ( [50, 200], {'step':1} ),
    'max_depth': ( [5, 15], {'step':1} ),
    'min_samples_leaf': ( [1, 3], {'step':1} ),
    'max_features': ( [1, x_train.shape[1]], {'step':1} )
}
hparams_float = { 'max_samples': ( [0.5, 1], {'step':0.01} ) }
model_args = {'random_state': 42}

opt_study = optuna_study(sklearn.ensemble.RandomForestRegressor, 
                     x_train, 
                     y_train,
                     study='new',
                     study_name=f'RF_with_cv',
                     direction='minimize',
                     )

opt_study.set_objective_conditions(
    hparams_cat = hparams_cat,
    hparams_int = hparams_int,
    hparams_float = hparams_float,
    model_locals = model_args,
    mode='cv',
    cv_scorers='rmse',
    cv_train_weight= 0,
    )
```


---

### method: `optimize()`

Executes the hyperparameter optimization process using the configured objective function. Supports various stopping criteria including fixed number of trials, score gradient convergence, and dynamic hyperparameter range expansion. Optionally exports checkpoints periodically as `.pkl` files.

**Parameters**

- `n_trials`: `int`  
    Number of trials to run per optimization iteration.

**Default Parameters:**

- `stop_criterium = n_trials`: `str`
    Criterion to stop optimization:  
        - `'n_trials'`: Stop after specified number of trials.  
        - `'best_score_gradient'`: Stop if score improvement falls below `score_gradient`.  
        - `'flex_range'`: Dynamically expand hyperparameter ranges when best values hit boundaries.

- `checkpoint  = None`: `int`
    Number of trials per checkpoint export.

- `checkpoint_path = None`: `str`
    File or directory path to save checkpoint files.

- `timeout = 10`: `int`
    Timeout in seconds per trial.

- `score_gradient = None`: `float`
    Threshold of score improvement for `'best_score_gradient'` stopping.

- `min_runs_score_gradient = 3`: `int`
    Minimum optimization iterations before applying score gradient check.

- `set_flex_protocol = None`: `dict`
    Protocol defining how to expand hyperparameter search ranges for `'flex_range'` stopping. Must be of the format:
```python
    set_flex_protocol = {
    'param_name': (protocol_for_min_range, protocol_value) ,(protocol_for_max_range, protocol_value)
    'param_name2': (protocol_for_min_range2, protocol_value2) ,(protocol_for_max_range2, protocol_value2)
    ...
    }
```

where `protocol_for_min_range` and `protocol_for_max_range` can be:

- `'add'`: The `protocol_value` will adds (+) the maximum or minimum of the search range when the hyperparameter reaches it. Its `value` must be numeric;
- `'mult'`: The `protocol_value` will multiply (*) the maximum or minimum of the search range when the hyperparameter reaches it. Its `value` must be numeric.

**Examples**

```python
opt_study.optimize(n_trials=100)

opt_study.optimize(
    n_trials=30,
    stop_criterium='best_score_gradient',
    score_gradient=0.01,
    min_runs_score_gradient=3
)

opt_study.optimize(
    n_trials=30,
    stop_criterium='flex_range',
    set_flex_protocol={
        'alpha': [('add', 0.01), ('mult', 2)],
        'max_depth': [('add', 1), ('add', 2)]
    }
)
```

```python
set_flex_protocol = {
    #Float:
    'C': (('mult', 0.1), ('add', 1000)),
    'gamma': (('mult', 0.1), ('add', 100)),
    'epsilon': (('mult', 0.1), ('add', 1)),
}
opt_study.optimize(
    n_trials=3000,
    stop_criterium='flex_range',
    checkpoint=500,
    checkpoint_path=f'./study/...',
    set_flex_protocol=set_flex_protocol,
)
```

