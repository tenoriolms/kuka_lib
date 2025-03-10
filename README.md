# 📚 API Reference

## __input_type_validation.py:__input_type_validation

**Parâmetros:** `func`

**Descrição:**

```
Decorator to validate the input types of a function.
The input types are defined in the function signature.

The function signature must have a specific `_input_type_dict` default argument.
The `_input_type_dict` argument must be a list of types like:
`  _input_type_dict = {
            'variable_name': type, 
            ...
            }
`
If theres no constraint for a variable, use `object` as the type.
```

--------------------------------------------------

## __input_type_validation.py:__type_validation

**Parâmetros:** `input_dict`

--------------------------------------------------

## _plt_hist_of_columns.py:plt_hist_of_columns

**Parâmetros:** `df, _input_type_dict={'df': type(pd.DataFrame())}`

**Descrição:**

```python
Plot a histogram for each numeric column of a 
dataframe in a more organized way than just calling pandas hist()
```

--------------------------------------------------

## cat_cols.py:str2int_hot_encoder

**Parâmetros:** `df, columns='all', _input_type_dict={'df': pd.DataFrame, 'columns': (str, list, tuple)}`

--------------------------------------------------

## cat_cols.py:str2int_simple_encoder

**Parâmetros:** `df, columns='all', _input_type_dict={'df': pd.DataFrame, 'columns': (str, list, tuple)}`

--------------------------------------------------

## cat_plot.py:plot_confusion_matrix

**Parâmetros:** `y_true, y_pred, labels=None, cmap='Blues', figsize=(6, 5), _input_type_dict={'y_true': (np.ndarray, list), 'y_pred': (np.ndarray, list), 'labels': (list, tuple, None), 'cmap': str, 'figsize': (tuple, list)}`

--------------------------------------------------

## categorization.py:classification_metrics

**Parâmetros:** `y_real, y_pred, _input_type_dict={'y_real': np.ndarray, 'y_pred': np.ndarray}`

--------------------------------------------------

## get_key.py:get_key

**Parâmetros:** `val, my_dict, default=None, _input_type_dict={'val': object, 'my_dict': dict, 'default': object}`

**Descrição:**

```python
Given a certain "value" in a dictionary, what is the "key" associated with it?

Returns the key associated with a specific value in a dictionary.

Args:
    val: The value to search for in the dictionary.
    my_dict: The dictionary to search in.
    default: The value to return if the value is not found (default: None).
    
Returns:
    The key associated with the value if found, otherwise the default value.
    
Examples:
    >>> get_key(2, {'a': 1, 'b': 2, 'c': 3})
    'b'
    >>> get_key(5, {'a': 1, 'b': 2, 'c': 3}, "Not found")
    'Not found'
```

--------------------------------------------------

## imp_exp_pkl.py:export_pkl

**Parâmetros:** `variable, path='default', _input_type_dict={'variable': object, 'path': str}`

**Descrição:**

```python
This function exports a variable of python as pickle file to the "path".
"path" NEEDS contain the name of the file.
```

--------------------------------------------------

## imp_exp_pkl.py:import_pkl

**Parâmetros:** `path='default', _input_type_dict={'path': str}`

**Descrição:**

```python
This function returns a an variable after define it according to a pickle file in "path".
"path" NEEDS contain the name of the file.
```

--------------------------------------------------

## num_cols.py:Zscores

**Parâmetros:** `df_for_scaled, df_reference, _input_type_dict={'df_for_scaled': pd.DataFrame, 'df_reference': pd.DataFrame}`

**Descrição:**

```python
Escalonar cada coluna de um DataFrame utilizando o "z score"
```

--------------------------------------------------

## num_cols.py:Zscores_with_param

**Parâmetros:** `df_real, dict_params, _input_type_dict={'df_real': pd.DataFrame, 'dict_params': dict}`

**Descrição:**

```python
Aplicar o escalonamento zscore para cada coluna de um DataFrame a partir dos parametros 
(média e desvio padrão)

If the variable is an "np.array", its necessary to convert it to a "DataFrame".

Example of dict_params:
dict_params = {
  'columns_dataframe_name1': {'mean':'mean_value1', 'std':'std_value1'},
  'columns_dataframe_name2': {'mean':'mean_value2', 'std':'std_value2'},
  'columns_dataframe_name3': {'mean':'mean_value3', 'std':'std_value3'}
}
```

--------------------------------------------------

## num_cols.py:normalize

**Parâmetros:** `df_for_norm, df_reference, _input_type_dict={'df_for_norm': pd.DataFrame, 'df_reference': pd.DataFrame}`

**Descrição:**

```python
Normalizar cada coluna de um DataFrame
```

--------------------------------------------------

## num_cols.py:undo_Zscores

**Parâmetros:** `df_scaled, df_reference, _input_type_dict={'df_scaled': pd.DataFrame, 'df_reference': pd.DataFrame}`

**Descrição:**

```python
desfazer o escalonamento realizado para cada coluna de um DataFrame utilizando
o "z score"
```

--------------------------------------------------

## num_cols.py:undo_Zscores_with_param

**Parâmetros:** `df_scaled, dict_params, _input_type_dict={'df_scaled': pd.DataFrame, 'dict_params': dict}`

**Descrição:**

```python
desfazer o escalonamento realizado para cada coluna de um DataFrame utilizando
o "z score" a partir dos parametros

If the variable is an "np.array", its necessary to convert it to a "DataFrame".

Example of dict_params:
dict_params = {
  'columns_dataframe_name1': {'mean':'mean_value1', 'std':'std_value1'},
  'columns_dataframe_name2': {'mean':'mean_value2', 'std':'std_value2'},
  'columns_dataframe_name3': {'mean':'mean_value3', 'std':'std_value3'}
}
```

--------------------------------------------------

## num_cols.py:undo_normalize

**Parâmetros:** `df_normalized, df_reference, _input_type_dict={'df_normalized': pd.DataFrame, 'df_reference': pd.DataFrame}`

**Descrição:**

```python
desfazer a normalização realizada para cada coluna de um DataFrame
```

--------------------------------------------------

## regression.py:c_coeff

**Parâmetros:** `v_real='class numpy.ndarray', v_pred='class numpy.ndarray'`

**Descrição:**

```python
Coeficiente proposto por Wessling et al (1997) (https://doi.org/10.1016/0376-7388(93)E0168-J)

The neural network works predictively if C is smaller than 1. For C=l, the
predicted permeability for an unknown polymer would be  equal to the average
permeability of all polymers presented in the set (which is, in fact, useless).
```

--------------------------------------------------

## regression.py:display_score

**Parâmetros:** `m, x_train, x_test, y_train, y_test, delog_y=False, base=10, untransform_y='Zscore or normalize', dict_params_transform='dicionario parametros', _input_type_dict={'m': object, 'x_train': np.ndarray, 'x_test': np.ndarray, 'y_train': np.ndarray, 'y_test': np.ndarray, 'delog_y': bool, 'base': (int, float), 'untransform_y': str, 'dict_params_transform': dict}`

**Descrição:**

```python
função para avaliar RMSE, R2 e OOB_score

Exemplo de dict_params_transform para Zscore:
dict_params_transform = {'mean':'mean_value_for_target_variable', 'std':'std_value_for_target_variable'}
Exemplo de dict_params_transform para normalize:
IMPLEMENTAR AINDA
```

--------------------------------------------------

## regression.py:mape

**Parâmetros:** `v_real, v_pred`

--------------------------------------------------

## regression.py:predictions_separate_by_a_variable

**Parâmetros:** `model, x_train, x_test, y_test, variable_in_original_databank, original_databank_train, original_databank_test, variable_subgroup='all', delog_y=False, base=10, untransform_y='Zscore or normalize', dict_params_transform='dicionario parametros', _input_type_dict={'model': object, 'x_train': np.ndarray, 'x_test': np.ndarray, 'y_test': np.ndarray, 'variable_in_original_databank': str, 'original_databank_train': pd.DataFrame, 'original_databank_test': pd.DataFrame, 'variable_subgroup': (list, str), 'delog_y': bool, 'base': (int, float), 'untransform_y': str, 'dict_params_transform': dict}`

**Descrição:**

```python
DEFINIÇÕES E HIPÓTESES:
Denomina-se databank original como aquele cujos valores não passaram por
transformações (encoders, padronização, normalização etc.), que são
iguais/semelhantes ao escrito na fonte de referência.
A conexão entre os conjuntos de teste/treino e os databanks originais são
estabelecidos pelos índices dos mesmos.

model = modelo
x_train = conjunto de treino com variáveis de entrada
x_test = conjunto de teste com variáveis de entrada
y_test = conjunto de teste com variáveis alvo
variable_in_original_databank = COLUNA CATEGÓRICA cujos valores ÚNICOS
                                servirão como referência para separar as
                                predições
original_databank_train = databank original, com valores originais NÃO
                            escalonados e sem transformações do conjunto de
                            treino
original_databank_test = databank original, com valores originais NÃO
                        escalonados e sem transformações do conjunto de
                        teste
subgroup = COLUNA CATEGÓRICA cujos valores ÚNICOS servirão como referência
            para separar as predições em subgrupos (OPCIONAL)

Exemplo de dict_params_transform para Zscore:
dict_params_transform = {'mean':'mean_value_for_target_variable', 'std':'std_value_for_target_variable'}
Exemplo de dict_params_transform para normalize:
IMPLEMENTAR AINDA
```

--------------------------------------------------

## regression.py:r2

**Parâmetros:** `v_real, v_pred`

--------------------------------------------------

## regression.py:rmse

**Parâmetros:** `v_real, v_pred`

--------------------------------------------------
