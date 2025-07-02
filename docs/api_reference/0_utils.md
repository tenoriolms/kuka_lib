<h1>Utilities for coding</h1>




--------------------------------------------------------------------------------------



## _input_function_validation

```
decorator utils._input_function_validation()
```

Decorator with input arguments.

A decorator used to validate function inputs using a user-defined validation protocol. It ensures input types and values match the expected constraints prior to executing the target function.

**Parameters:**

* `_input_val_dict`: dict

Dictionary that will carry the entire validation protocol and must be like:
```python
_input_type_dict = {
    'variable_name1': ( var_type->(str, list, tuple), validation_mode->str, validation_parameters ),
    'variable_name2': ( var_type->(str, list, tuple), validation_mode->str, validation_parameters ),
    ...
    }
```

The `variable_name` key must be a string and its value must be a list or tuple. If theres no type constraint for a variable, use `object` as the type.

The variable `validation_mode` and respective `validation_parameters` can have the values of:

* `None or 'none'`: There is no restriction on the parameter value.
    * `validation_parameters` = Anything. Will not be read. Can be occulted.
* `'pattern'`: The parameter value must be exactly one of a list of values.
* `validation_parameters` = A list/tuple of exact valid values. Ex: `['a', 'b', 'c'] or [1, 3, 5, 6] or ['a', 3, {'a':1}]`.
* `'range'`: The parameter value has minimum and maximum values. It only works with numeric parameters.
    * `validation_parameters` = A list/tuple with the format `[min, max]`. `np.inf` can be used.
* `'length'`: The parameter is a list, tuple or string with a given length.
    * `validation_parameters` = Int. Length of the tuple/list. The `len()` function is applied.

**Returns:**

* `none`

**Raises:**

- `ValueError`: If `_input_val_dict` is malformed or contains invalid configuration.
- `AssertionError`: If an input value does not match the expected type or fails the validation rules.

**Examples:**

```python
> import numpy as np
> from kuka.utils import _input_function_validation
>
> _input_val_dict = {
>     'a': (str, None)
>     'b': ((list, tuple), 'length', 4)
>     'c': ((int, float), 'range', [0, 10])
>     'd': ((int, float), 'range', [5, np.inf])
>     'e': (str, 'pattern', ['opt1', 'opt2', 'opt3'])
>     }
> 
> @_input_function_validation(_input_val_dict)
> def function(a, b, c=1, d=42, e='opt1'):
>     ...
>     return ...
```



--------------------------------------------------------------------------------------



## _is_valid_path
```
function utils._is_valid_path()
```

Checks whether a given string is a valid path to an EXISTING file or directory..

**Parameters:**

* `path_str`: type, valid values

    Parameter description.

**Default Parameters:**

* `type_path = 'file or dir'`: str, "file" or "dir"

    Parameter description.

* `check_only_syntax = False`: type, valid values

    Parameter description.


**Returns:**
* `True or False`: bool
    `True` if `path_str` is a valid and existing path of the specified type, `False` otherwise.

**Raises:**

* `AssertionError`:

    If `type_path` string is neither a "file" nor a "dir".


**Notes:**

This function validates whether `path_str` is a properly formatted string that can represent a file or directory path on the current operating system. It checks:
- That it is a non-empty string
- That it does not contain invalid characters (especially on Windows)
- That backslashes are used correctly (e.g., allows network paths like \\\\server)
- That it points to an existing file or directory, depending on `type_path`

**Examples:**

```python
> __is_valid_path("data/model.pkl", type_path="file")
> __is_valid_path("data/", type_path="dir")
```



--------------------------------------------------------------------------------------



## plot

### _adjust_obj_to_ref
```
function utils.plot._adjust_obj_to_ref()
```

Automatically adjusts matplotlib text-like objects to avoid overlap or excessive size compared to reference elements. Applies a sequence of correction steps like rotation, font size scaling, abbreviation, repositioning, or text wrapping to make labels or texts fit within the visual bounds of a figure.

**Parameters**

- `fig` (`matplotlib.figure.Figure`):  
    The figure containing the objects to be adjusted. Used to calculate bounding boxes.

- `obj_list` (`list`):  
    List of matplotlib objects to be adjusted. Objects must implement `.get_window_extent()`.

```
	Ex: 
	list(text_list) = [Text(), Text(), Text(), ...]
	list(ax.texts), ax.get_xticklabels()
```

- `ref_list` (`list`):  
    List of reference elements used to compare dimensions or check overlap. Ex.: `[ax]`.

- `dimension` (`str`):  
    Direction of dimensional constraint: `'w'` for width or `'h'` for height.

- `steps` (`list` of `str`):  
    Sequence of adjustment strategies to attempt, in order. Available options:
    - `'try_rotate'`: Rotate text incrementally. Has "get_rotation()" and "set_rotation()" methods;
    - `'decrease_fontsize'`: Scale down font size. Has "get_fontsize()" and "set_fontsize()" methods;
    - `'reset_fontsize'`: Restore original font size. Has "set_fontsize()" method;
    - `'remove_label'`: Remove all labels/texts. Has "set_text()" method;
    - `'abbreviate_or_remove_tick_labels'`: Replace tick labels with abbreviated symbols. Is a "ax.get_xticklabels()" or "ax.get_yticklabels()" matplotlib object. works only if the ticklabels obey the common position for their respective axes. (x, 0) for the x_axis and (0, y) for the y_axis;
    - `'abbreviate_or_remove'`: Replace text content with abbreviations or remove them. Has ".get_text()", "set_text()" and ".get_position()" method;
    - `'move_until_no_overlaps'`: Move objects until no overlap is detected. Has ".set_position()" method;
    - `'wrap_until_fit'`: Wrap long text into multiple lines. Has ".get_text()" and ".set_text()" methods.

**Default Parameters**

- `threshold = 0.8` (`float`):  
    Fraction of total space occupied by `obj_list` relative to `ref_list` before adjustments are applied.

- `rotate_max = 90` (`float`):  
    Maximum rotation angle (degrees) for `'try_rotate'` step.

- `rotate_rate = 5` (`float`):  
    Incremental rotation applied per attempt in `'try_rotate'`.

- `fontsize_max = 7.` (`float`):  
    Threshold under which font size will not be decreased.

- `fontsize_rate = 0.9` (`float`):  
    Multiplicative factor to reduce font size during `'decrease_fontsize'`.

- `abbreviate_list = ['letters']` (`list` or `tuple`):  
    Custom abbreviations to use in abbreviation steps. Defaults to letter-based abbreviations.

- `tick_ax = None` (`matplotlib.axes.Axes` or `None`):  
    Axis object required for `'abbreviate_or_remove_tick_labels'` step.

- `move_rate = (None, None)` (`tuple`):  
    (x, y) amount to translate objects during `'move_until_no_overlaps'`.

- `wrap_rate = 2` (`int`):  
    Width decrement step used in `'wrap_until_fit'`.

- `wrap_maxlines = 3` (`int`):  
    Maximum number of lines allowed in wrapped text.

- `_plot_ref = False` (`bool`):  
    Whether to plot reference bounding boxes for debugging.

- `_plot_obj = False` (`bool`):  
    Whether to plot object bounding boxes for debugging.

**Returns**
- `None`:  
  The function operates in-place, modifying the input objects directly.

**Examples**
```python
fig, ax = plt.subplots()
bars = ax.bar(['Very very long label A', 'Extremely long label B'], [3, 5])
fig.canvas.draw()
ticklabels = ax.get_xticklabels()
```

```python
# Rotate and shrink labels to fit
_adjust_obj_to_ref(fig, ticklabels, bars, dimension='w',
                   steps=['try_rotate', 'decrease_fontsize'], threshold=0.9)
```

```python
# Abbreviate labels if needed
_adjust_obj_to_ref(fig, ticklabels, bars, dimension='w',
                   steps=['abbreviate_or_remove_tick_labels'],
                   abbreviate_list=['A', 'B'], tick_ax=ax)
```

```python
# Wrap long text into multiple lines
texts = [ax.text(0.5, 0.95, 'A very long title', ha='center', transform=ax.transAxes)]
_adjust_obj_to_ref(fig, texts, [ax], dimension='w',
                   steps=['wrap_until_fit'], wrap_rate=3, wrap_maxlines=3)

```



--------------------------------------------------------------------------------------



### _plot_bboxes

```python
function utils.plot._plot_bboxes()
```

Plots bounding boxes on a matplotlib figure using figure-relative coordinates (0–1).

This function helps visualize and debug layout issues by drawing rectangular outlines corresponding to graphical elements such as tick labels, texts, or axes objects. It converts pixel-based bounding boxes to figure fraction coordinates and overlays them on the figure.

**Parameters:**

- `bbox_list`: list or tuple of matplotlib.transforms.Bbox, valid values: —  
  A list of bounding boxes to be drawn, given in pixel coordinates (usually from `.get_window_extent(renderer)`).

- `fig`: matplotlib.figure.Figure, valid values: —  
  The target figure on which bounding boxes will be drawn.

**Default Parameters:**

- `color = 'red'`: str, valid values: —  
  Color of the bounding box borders.

- `linewidth = 1`: int or float, valid values: range [0, ∞)  
  Thickness of the rectangle edges.

- `linestyle = '-'`: str 
  Style of the rectangle edges (e.g., '-', '--', ':', '-.').

**Returns:**

- `rect_list`: list of matplotlib.patches.Rectangle  
  List of the rectangle objects added to the figure.

**Examples:**

```python
    fig, ax = plt.subplots()
    bars = ax.bar(['A', 'B', 'C'], [5, 7, 4])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
```
```python
    bbox_list = [bar.get_window_extent(renderer) for bar in bars]
    _plot_bboxes(bbox_list, fig, color='green', linewidth=2, linestyle='--')
```
```python
    # Also works with tick labels or text objects:
    tick_bboxes = [label.get_window_extent(renderer) for label in ax.get_xticklabels()]
    _plot_bboxes(tick_bboxes, fig, color='blue')
```



--------------------------------------------------------------------------------------



### _plot_text

```python
function utils.plot._plot_text()
```

Plot a text in matplotlib. Useful to show some simple LaTeX codes.

**Parameters:**

- `text`: str
    Text string to be plotted.

**Default Parameters:**

- `_fontsize = 12`: float or int, valid values: range [0, ∞)
    Font size for the plotted text.

**Returns:**

- tuple of (`matplotlib.figure.Figure`, `matplotlib.axes.Axes`)  
    The figure and axes objects containing the plotted text.

**Example:**

```python
fig, ax = _plot_text("Hello World", _fontsize=14)
fig.show()
```


