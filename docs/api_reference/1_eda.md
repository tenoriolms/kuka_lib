<h1> EDA - (Exploratory Data Analysis)</h1>



--------------------------------------------------------------------------------------


## plot_columns_dist

```python
function plot.plot_columns_dist()
```

Plot a grid of distributions for each column in a DataFrame.

This function automatically detects the type of each column and chooses the appropriate plot:
- Numerical (float-compatible) columns are displayed using histograms.
- Non-numerical (object/categorical) columns are shown as bar plots with value counts.

It organizes all plots into a structured subplot grid, and includes automatic font size adjustments
and label abbreviation to avoid label overlaps, especially for categorical variables.

**Parameters:**

- `df`: `pd.DataFrame`
    Dataframe containing the columns to be plotted.

**Default Parameters:**

- `ncols = 3`: `int`, valid values: range [0, ∞)  
    Number of columns in the subplot grid layout.

- `plotsize = plt.rcParams['figure.figsize'] * [0.5, 0.35]`: `list` or `tuple` (length=2), valid values: positive numbers  
    Base plot size (width, height) in inches for each subplot. The total figure size is scaled by (plotsize * [ncols, nrows]).
    
**Returns:**

- `fig`: matplotlib.figure.Figure  
    The generated figure.

- `ax`: numpy.ndarray of matplotlib.axes._subplots.AxesSubplot  
    Array of axes corresponding to each subplot.

**Examples:**

```python
from kuka.eda import plot_columns_dist
import seaborn as sns

df = sns.load_dataset('penguins')

fig, ax = plot_columns_dist(
    df,
    ncols=2,
    plotsize=[4, 3]
)

fig.show()
```



--------------------------------------------------------------------------------------



## plot_dist

```python
function plot.plot_dist()
```

Plot the distribution of a variable using bar plots or histograms, with support for category segmentation, normalization, and extensive customization options.

This function is useful for visually exploring the frequency distribution of a variable, optionally segmented by a categorical column (`by`). It supports different layout modes such as stacked bars, grouped bars, or separated subplots.

**Parameters** 

- `df`: `pandas.DataFrame`
    Input DataFrame containing the data to be plotted.

- `variable`: `str`  
    Name of the column to be plotted.

- `mode`: `str`  
    Plotting mode:  
        - `'bar'` : for categorical variables (bar chart)  
        - `'hist'`: for numeric variables (histogram).

**Default Parameters:**

- `by = None`: `str` or `None` 
    Name of the grouping column for segmentation (used to separate or color the distributions).

- `by_mode = 'stacked'`: `str` {'stacked', 'grouped', 'subploted'}
    Layout style when grouping is applied:  
        - `'stacked'`   : stacked bars  
        - `'grouped'`   : side-by-side grouped bars  
        - `'subploted'` : one subplot per category

- `alpha = 0.5`: `float`
    Transparency level of the bars (from 0 to 1).

- `colors_reference = ['b','g','r','c','m','y','k']`: `list` or `tuple`  
    List of colors to assign to `by` categories (used if `dict_colors_of_variable_by` is not provided).

- `figsize = None`: `list` or `tuple` of length 2
    Size of the figure in inches: `[width, height]`.

- `fontsize = None`: `int` or `float` 
    Global font size for the plot.

- `labels = None`: `None`, `'top'`, `'center' `
    If not None, adds value labels to the bars:  
        - `'top'`    : place above the bars  
        - `'center'` : place inside the bars

- `labels_format = None`: `str`  
    Format string for bar labels (e.g., '.0f', '.1%', etc.).

- `labels_padding = [0, 0]`: `list` or `tuple` of length 2
    Offset for labels in points: `[x_offset, y_offset]`.

- `labels_threshold = 0.85`: `float`
    Minimum proportion of overlap allowed for the label to remain visible.

- `norm = None`: `bool`
    If True, normalize bar heights so they sum to 1. Useful for comparing distributions.

- `bar_width = 0.8`: `float`
    Width of the bars in `'bar'` mode (between 0 and 1).

- `hist_bins = 10`: `int`
    Number of bins for the histogram.

- `hist_ticklabels_format = '.2f'`: `str` 
    Format string for x-axis tick labels in histogram mode.

- `hist_grouped_offset = [None, None]`: `list` or `tuple` of 2 elements
    Offset for each group in grouped histogram mode: `[x_offset, y_offset]`.

- `hist_x_log_scale = False`: `bool`
    If True, applies log10 scale to the x-axis before plotting.

- `hist_average_line = False`: `bool`
    If True, adds a smoothed KDE (kernel density estimate) curve over the histogram.

- `hist_average_line_kargs = {'linewidth':1, 'alpha':0.25, 'edgecolor':'black', 'color':'black'}`: `dict`
    Additional keyword arguments for the KDE curve.

- `external_fig_and_ax = None`: `None` or `tuple(Figure, Axes)`
    If provided, uses the given Matplotlib `Figure` and `Axes` for plotting.

- `dict_colors_of_variable_by = None`: `dict` or `None`
    Custom color mapping for each category in the `by` column.

- `verbose = None`: `bool` 
    If True, prints the min and max values of the `variable`.

- `**bar_kargs`  
    Additional keyword arguments passed directly to `ax.bar()`.

**Returns**  
- `fig` (matplotlib.figure.Figure)  
    The created Matplotlib Figure object.

- `ax` (matplotlib.axes.Axes or numpy.ndarray of Axes)  
    The created Axes object(s). May be an array if `by_mode='subploted'`.

**Examples**

```python
import pandas as pd

df = pd.DataFrame({'Category': ['A', 'B', 'A', 'C', 'B'], 'Value': [10, 20, 30, 40, 50]})
```

```python
# Basic bar plot
plot_dist(df, variable='Category', mode='bar')
```

```python
# Histogram plot with custom bins
plot_dist(df, variable='Value', mode='hist', hist_bins=5)
```

```python
# Grouped bar plot by a second variable
plot_dist(df, variable='Category', mode='bar', by='Value', by_mode='grouped')
```



--------------------------------------------------------------------------------------



## plot_pairplot

```python
function plot.plot_pairplot()
```

Create a pairplot-style scatter plot matrix with optional grouping and KDE contours.

This function plots scatter plots for all combinations of the provided columns in a DataFrame, optionally grouped by a categorical column (`by`). It allows color and marker customization for each group and can overlay KDE contours.

**Parameters**

- `df` : `pandas.DataFrame`  
  Input dataset containing the columns to be plotted.

**Default Parameters:**

- `columns = ['all']` : `list`
    List of columns to include in the plot. If `['all']`, all numeric columns in the DataFrame are used.

- `by = None` : `str` or `None`  
    Column used to group data into different categories.  

- `colors_reference = ['black', '#023eff', '#ff7c00', '#1ac938', '#e8000b', '#8b2be2', '#9f4800', '#f14cc1', '#a3a3a3', '#ffc400', '#00d7ff']` : `list`  
    List of colors to use for each group in `by`.

- `dict_colors_of_variable_by = None` : `dict or None`  
    Dictionary mapping each group to a specific color.

- `zorder = None` : `list or None`
    Drawing order of groups. Groups listed later appear on top.

- `markers  = None` : `list or None`
    List of marker styles for each group.

- `fontsize = 10.` : `float`
    Font size for axis labels.

- `kargs_labels = {}` : `dict`  
    Additional keyword arguments for label formatting.

- `figsize = (6, 6)` : `tuple`  
    Figure size in inches.

- `wspace = 0.12` : `float`  
    Horizontal space between subplots.

- `hspace = 0.12` : `float`  
    Vertical space between subplots.

- `s = 10` : `int` 
    Marker size in the scatter plot.

- `tick_majorlabel = True` : `bool`  
    Whether to show major tick labels.

- `tick_labelsize = None` : `float or None`  
    Size of tick labels.

- `tick_labelformat = '{x:.4g}'` : `str`  
    Format string for tick labels.

- `margins = 0.08` : `float`  
    Margin added around plot limits.

- `show_kdeplots = None` : `int or None`  
    If set, overlays KDE contour plots at the specified z-order.

- `kdeplots_level = 3` : `int`  
    Number of contour levels in KDE plots.

- `show_kdeplots_of_by = False` : `bool`  
    Whether to plot KDE contours separately for each group in `by`.

- `kargs_kdeplots = {}` : `dict`  
    Additional keyword arguments passed to `sns.kdeplot`.

- `**kwargs_scatter`
    Additional keyword arguments. Passed to `scatter` for further customization.

**Returns**

- `fig` : `matplotlib.figure.Figure`  
  The generated figure object.

- `ax` : `numpy.ndarray` of `matplotlib.axes.Axes`  
  A 2D array of Axes containing the scatter plots.

**Examples**

```python
fig, ax = plot_plairplot(df, columns=['height', 'weight', 'age'], by='gender', s=20, show_kdeplots=2)
```
