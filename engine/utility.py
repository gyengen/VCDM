import os
import numpy as np
import pandas as pd
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from bokeh.embed import components
from collections import OrderedDict
from bokeh.models import ColumnDataSource, HoverTool, Range1d, LinearAxis

plot_height, plot_width = 250, 600


def delete_old_files(t):
    '''Deleting files in the pub folder.

    Args:
        t (str): How old files to be deleted in mitues

    Returns:
        os command

    '''

    # Generate linux command
    CMD = "find " + os.getcwd() + "/pub -mmin +" + t + " -delete; "

    # Create pub directory if it does not exist
    CMD += "mkdir -p " + os.getcwd() + '/pub'

    # Clear upload folder, files older than x minutes
    return os.system(CMD)


def allowed_file(fn):
    '''Checking the file extensions. Only csv file with hheaderis
        permitted at this stage of development. Excel formats could
        be also considered.

    Args:
        fn (str): Filename

    Returns:
        bool: True if extension is allowed, False otherwise.

    '''

    # A set of accapteble extension. At the moment only .csv
    EXT = set(['csv'])

    # Check file extension. Only csv allowed at this stage.
    return '.' in fn and fn.rsplit('.', 1)[1].lower() in EXT


def generate_results(fn, d, df):
    '''This function generates the results as a bokeh figure.

    Args:
        fn(strin): Filename for output PDF files
        d (dict): Generated by the vcdm engine.
        df (pandas dataframe): original data

    Returns:
        str: Bokeh embedded Javascript for the frontend template.

    '''

    # Create 3 subplots
    s1 = figure()
    s2 = figure(x_range=s1.x_range)
    s3 = figure(x_range=s1.x_range)

    # Create Data Source for plotting the data
    d1 = ColumnDataSource(pd.DataFrame({"x": d['time'], "y": d['shear']}))

    # First plot
    s1.line(x="x",
            y="y",
            source=d1,
            color="#0066b3",
            alpha=0.5,
            line_width=1.5)

    # 2nd plot
    for i, colour in enumerate(["#0066b3", "#117733", "#999933"]):

        # Generate label
        label = "{}th percentile={:.2f}Pa".format(d['percentile_value'][i][0],
                                                  d['percentile_value'][i][1])

        # Create Data Source for plotting the data
        d2 = ColumnDataSource(pd.DataFrame({"x": d['time'], "y": d['phi'][i]}))

        # Create lineplot
        s2.line(x="x",
                y="y",
                source=d2,
                color=colour,
                legend_label=label,
                line_width=2)

    # Create Data Source for plotting the data
    if len(df.columns) == 4:
        d3 = ColumnDataSource(pd.DataFrame({"x": d['time'],
                                            "y": d['turbidity'],
                                            "x2": np.array(df.iloc[:, 2]),
                                            'y2': np.array(df.iloc[:, 3])}))

    else:
        d3 = ColumnDataSource(pd.DataFrame({"x": d['time'],
                                            "y": d['turbidity']}))

    # Plotting the data
    s3.line(x="x",
            y="y",
            source=d3,
            color="#0066b3",
            legend_label='Turbidity',
            line_width=2)

    # If the input data has 4 columns
    if len(df.columns) == 4:

        # Plotting the data
        s3.line(x="x2",
                y="y2",
                source=d3,
                legend_label='Measured Turbidity',
                color="#117733",
                line_width=2)

    # Generate labels for the hover tool
    label = ['Applied Shear [Pa]',
             'Phi [tau, t]',
             'Turbidity [NTU]',
             'Measured Turbidity']

    # Create custom hover and add the hover to the plot
    s1.add_tools(HoverTool(tooltips=[('Time [s]', '@x'), (label[0], '@y')]))
    s2.add_tools(HoverTool(tooltips=[('Time [s]', '@x'), (label[1], '@y')]))

    if len(df.columns) == 4:
        s3.add_tools(HoverTool(tooltips=[('Time [s]', '@x'), (label[2], '@y'),
                                         ('Time [s]', '@x2'), 
                                         (label[3], '@y2')]))

    else:
        s3.add_tools(HoverTool(tooltips=[('Time [s]', '@x'),
                                         (label[2], '@y')]))

    # Setup the y labels
    s1.yaxis.axis_label = 'Applied Shear [Pa]'
    s2.yaxis.axis_label = 'phi [tau, t]'
    s3.yaxis.axis_label = 'Turbidity [NTU]'

    # Setup the x labels
    s1.xaxis.axis_label = 'Time [s]'
    s2.xaxis.axis_label = 'Time [s]'
    s3.xaxis.axis_label = 'Time [s]'

    # Transparent background
    s1.border_fill_color = None
    s1.background_fill_color = None

    s2.border_fill_color = None
    s2.background_fill_color = None

    s3.border_fill_color = None
    s3.background_fill_color = None

    # Remove grid
    s1.xgrid.grid_line_color = None
    s1.ygrid.grid_line_color = None

    s2.xgrid.grid_line_color = None
    s2.ygrid.grid_line_color = None

    s3.xgrid.grid_line_color = None
    s3.ygrid.grid_line_color = None

    s2.legend.background_fill_alpha = 0.0
    s3.legend.background_fill_alpha = 0.0

    # If the input data has 4 columns
    if len(df.columns) == 4:

        # make a grid
        grid = gridplot([s1, s2, s3],
                        ncols=1,
                        plot_width=plot_width,
                        plot_height=plot_height)

    else:

        # make a grid
        grid = gridplot([s1, s2, s3],
                        ncols=1,
                        plot_width=plot_width,
                        plot_height=plot_height)

    return components(grid)


def generate_turbidity_and_flow(df):
    '''Generating line plot for the input, offering a nice tool for
    previewing the data, checking for outliers.

    Args:
        df (Pandas dataframe): Raw data from the uploded file.

    Returns:
       str: Bokeh embedded Javascript for the frontend template.

    '''

    # If the input data has 4 columns
    if len(df.columns) == 4:

        # Custom hover poperties
        t_and_f = pd.DataFrame({"time": np.array(df.iloc[:, 0]),
                                "flow": np.array(df.iloc[:, 1]),
                                "time_turb": np.array(df.iloc[:, 2]),
                                "turb": np.array(df.iloc[:, 3])})

    else:

        # Custom hover poperties
        t_and_f = pd.DataFrame({"time": np.array(df.iloc[:, 0]),
                                "flow": np.array(df.iloc[:, 1])})

    # Geenrate source data for plotting
    d1 = ColumnDataSource(t_and_f)

    # Setup toolbox
    tools = "box_zoom,reset,pan,save"

    # Generate a histogram
    s1 = figure(plot_width=plot_width,
                plot_height=plot_height,
                tools=tools)

    # Create lineplot #1
    s1.line(x="time",
            y="flow",
            source=d1,
            color="#0066b3",
            legend_label='Flow',
            line_width=1)

    # Range for the first dataset
    s1.y_range = Range1d(t_and_f['flow'].min(), t_and_f['flow'].max())

    # Setup the label for y axis
    s1.yaxis.axis_label = str(df.keys()[1])

    # If the input data has 4 columns
    if len(df.columns) == 4:

        # Range for the 3nd dataset
        s1.extra_y_ranges = {'y2': Range1d(t_and_f['turb'].min(),
                                           t_and_f['turb'].max())}

        # Add 2nd axis
        s1.add_layout(LinearAxis(y_range_name='y2',
                                 axis_label=str(df.keys()[3])), "right")

        # Create lineplot #1
        s1.line(x="time_turb",
                y="turb",
                source=d1,
                color="#117733",
                y_range_name='y2',
                legend_label='Turbidity [NTU]',
                line_width=1)

        # Create custom hover and add the hover to the plot
        s1.add_tools(HoverTool(tooltips=[('Time [s]', '@time'),
                                         (str(df.keys()[1]), '@flow'),
                                         ('Time [s]', '@time_turb'),
                                         (str(df.keys()[3]), '@turb')]))

    else:

        # Create custom hover and add the hover to the plot
        s1.add_tools(HoverTool(tooltips=[('Time [s]', '@time'),
                                         (str(df.keys()[1]), '@flow')]))

        # Setup the label for y axis
        s1.yaxis.axis_label = str(df.keys()[1])

    # Setup the label for x axis
    s1.xaxis.axis_label = str(df.keys()[0])

    # Transparent background
    s1.border_fill_color = None
    s1.background_fill_color = None

    # Remove grid
    s1.xgrid.grid_line_color = None
    s1.ygrid.grid_line_color = None

    # Legend properties
    s1.legend.background_fill_alpha = 0.0
    s1.legend.location = "top_left"

    return components(s1)


def generate_histogram(df):
    '''Generating histogram for the input, offering a nice tool for
    previewing the data, checking for outliers.

    Args:
        df (Pandas dataframe): Raw data from the uploded file.

    Returns:
       str: Bokeh embedded Javascript for the frontend template.

    '''

    # Setup toolbox
    tools = "box_zoom,reset,save"

    # Generate a histogram
    p = figure(plot_width=plot_width,
               plot_height=plot_height,
               title="Flow Histogram",
               tools=tools)

    # Slice the histogram data from the datafram
    hist_data = np.array(df.iloc[:, 1])

    # Generate the histogram without NaNs
    hist, edges = np.histogram(hist_data[~np.isnan(hist_data)])

    # Custom hover poperties
    hist_df = pd.DataFrame({"column": hist,
                            "left": edges[:-1],
                            "right": edges[1:]})

    hist_df["interval"] = ["%.2f to %.2f" % (left, right)
                           for left, right
                           in zip(hist_df["left"], hist_df["right"])]

    # Generate data source object for plotting the data
    src = ColumnDataSource(hist_df)

    # Create a plot
    p.quad(top="column", bottom=0, left="left", right="right",
           fill_color="#0066b3", line_color=None, alpha=.5, source=src)

    # Create custom hover
    h = HoverTool(tooltips=[('Interval', '@interval'), ('Count', '@column')])

    # Add the hover to the plot
    p.add_tools(h)

    # Setup the label for x axis
    p.xaxis.axis_label = str(df.keys()[1])

    # Setup the label for y axis
    p.yaxis.axis_label = 'Count Number'

    # Transparent background
    p.border_fill_color = None
    p.background_fill_color = None

    # Remove grid
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    # Generate the div and java script for the HTML template
    return components(p)


def generate_stats(df):
    '''Describing statistics based on the raw input data, providing a
    quick exploratory data analysis.

    Args:
        df (Pandas dataframe): Raw data from the uploded file.

    Returns:
        OrderedDict: Containing the basics stats of the data.

    '''

    # Extract the data from describe method and save it to a dict
    stats = df.describe().round(decimals=4).T.to_dict("list")

    # Add the header
    stats['header'] = list(df.describe().columns.values)

    return OrderedDict(stats.items())


def generate_xls(fn, res, raw):
    '''Create an xls file as a downloadable content.

    Args:
        res (dict): Generated by the vcdm engine.
        raw (Pandas dataframe): Raw data from the uploded file.
    Returns:
        str: path and name of the new xls file

    '''

    # Save few columns from the original data
    if len(raw.columns) == 2:

        # Create a dictionary for the output file
        df = pd.DataFrame({"t": res['time'],
                           "tau_a": res['shear'],
                           "turb": res['turbidity']})

    elif len(raw.columns) == 3:
        _, _, a1 = raw.values.T
        header = raw.keys()

        # Create a dictionary for the output file
        df = pd.DataFrame({"t": res['time'],
                           "tau_a": res['shear'],
                           "turb": res['turbidity'],
                           str(header[-1]): a1})

    elif len(raw.columns) == 4:
        _, _, a1, a2 = raw.values.T
        header = raw.keys()

        # Create a dictionary for the output file
        df = pd.DataFrame({"t": res['time'],
                           "tau_a": res['shear'],
                           "turb": res['turbidity'],
                           str(header[-2]): a1,
                           str(header[-1]): a2})

    # Set the index
    df = df.set_index("t")

    # Create a filename
    filename = str(fn.split('.')[0]) + ".xlsx"

    # Generate xls file
    df.to_excel(filename, sheet_name='time_series')

    return filename
