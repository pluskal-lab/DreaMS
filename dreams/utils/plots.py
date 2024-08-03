import collections as coll
import pandas as pd
import numpy as np
import statistics as stat
import matplotlib.pyplot as plt
import seaborn as sns
import pylab
import matplotlib as mpl
# import termplotlib as tpl
from dreams.definitions import FIGURES


# def terminal_hist(arr):
#     counts, bin_edges = np.histogram(arr)
#     fig = tpl.figure()
#     fig.hist(counts, bin_edges, orientation='horizontal', force_ascii=True)
#     fig.show()


def color_generator(n_colors, cmap='plotly'):
    if cmap == 'plotly':  # https://stackoverflow.com/questions/41761654/plotly-where-can-i-find-the-default-color-palette-used-in-plotly-package
        return iter((n_colors // 10 + 1) * [
            (0.38823529411764707, 0.43137254901960786, 0.9803921568627451, 1.0),
            (0.0, 0.8, 0.5882352941176471, 1.0),
            (0.8392156862745098, 0.15294117647058825, 0.1568627450980392, 1.0),
            (0.09019607843137255, 0.7450980392156863, 0.8117647058823529, 1.0),
            (0.7372549019607844, 0.7411764705882353, 0.13333333333333333, 1.0),
            (0.5803921568627451, 0.403921568627451, 0.7411764705882353, 1.0),
            (0.5490196078431373, 0.33725490196078434, 0.29411764705882354, 1.0),
            (0.8901960784313725, 0.4666666666666667, 0.7607843137254902, 1.0),
            (0.4980392156862745, 0.4980392156862745, 0.4980392156862745, 1.0),
            (0.9372549019607843, 0.3333333333333333, 0.23137254901960785, 1.0),
            (0.0859375, 0.46484375, 0.3671875, 1.0),
            (0.84, 0.8, 0.33, 1.0)
        ][:n_colors])
    elif cmap == 'hsv':
        return (pylab.get_cmap(cmap)(1. * i / n_colors) for i in range(n_colors))
    elif cmap == 'pastel':
        return iter((n_colors // 10 + 1) * [
            (0.3125    , 0.6484375 , 0.56640625, 1.0),
            (0.94921875, 0.64453125, 0.72265625, 1.0),
            (0.25390625, 0.44140625, 0.73046875, 1.0),
       ][:n_colors])
    else:
        raise ValueError(f'Invalid cmap "{cmap}".')


def rgb_to_hex(r, g, b):
    r = int(256 * r) if isinstance(r, float) else r
    g = int(256 * g) if isinstance(g, float) else g
    b = int(256 * b) if isinstance(b, float) else b
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)


def get_palette(cmap='plotly', reversed_order=False, as_hex=False):
    if cmap == 'nature':
        return get_nature_hex_colors()
    palette = list(color_generator(12, cmap=cmap))
    if as_hex:
        palette = [rgb_to_hex(int(256 * c[0]), int(256 * c[1]), int(256 * c[2])) for c in palette]
    if reversed_order:
        return list(reversed(palette))
    return palette


def init_plotting(figsize=(6, 2), font_scale=0.95, style='whitegrid', cmap='plotly', font=None, legend_outside=False):
    # Set default figure size
    plt.show()  # Does not work without this line for some reason
    sns.set(rc={'figure.figsize': figsize})
    # Set default style and  font scale
    sns.set_style(style)
    sns.set_context('paper', font_scale=font_scale)
    sns.set_palette(get_palette(cmap))
    if not font:
        mpl.rcParams['svg.fonttype'] = 'none'
        # mpl.rcParams['pdf.fonttype'] = 42  # For Adobe Illustrator
        # mpl.rcParams['ps.fonttype'] = 42  # For Adobe Illustrator
    else:
        mpl.rcParams['font.family'] = font
    if legend_outside:
        plt.legend(loc='upper left', bbox_to_anchor=(1, 0.5))


def assign_colors(x):
    return dict(zip(x, [rgb_to_hex(*c) for c in sns.color_palette("Spectral", as_cmap=False, n_colors=len(x))]))


def get_nature_hex_colors(extended=True):
    palette = ['#2664BF', '#34A89A', '#F69CA9', '#FBD399', '#AD95D1', '#FEA992']
    if extended:
        palette += ['#AB8D8B', '#A8A9AB', '#6A4C93', '#C7EFCF', '#00CED1', '#FF6F61']
    return palette


def save_fig(name, dir=FIGURES, dpi=None, transparent=True):
    plt.savefig((dir / name) if dir is not None else name, bbox_inches='tight', pad_inches=0.05, dpi=dpi,
                transparent=transparent)


def distr_density(values,
                  domain=None,
                  show_mean=True,
                  show_median=False,
                  title=None):

    values = pd.Series(values).astype('float')

    if domain is not None:
        # Filter out values not belonging to the given domain
        values = values.loc[values.map(lambda e: e >= domain[0] and e <= domain[1])]
        if values.empty:
            print('ERROR: No values in the specified domain.')
            return

    min_elem = min(values)
    max_elem = max(values)
    mean = stat.mean(values)
    median = stat.median(values)
    stdev = stat.stdev(values)
    modus = coll.Counter(values).most_common(1)[0][0]

    sns.histplot(values, kde=True)

    if show_mean:
        plt.axvline(x=mean, ymin=0.0, ymax=0.97, ls='--', color='red',
                    label='Mean ({:.2f})'.format(mean))
    if show_median:
        plt.axvline(x=median, ymin=0.0, ymax=0.97, ls='--', color='green',
                    label='Median ({:.2f})'.format(median))

    # Place legend outside
    plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
    plt.title(title)

    stat_string = '| Mean: {:.2f} | Stdev: {:.2f} | Median: {:.2f} | Modus: {:.2f}'\
                  ' | Min: {:.2f} | Max: {:.2f} |'.format(mean, stdev, median,
                  modus, min_elem, max_elem)
    print(len(stat_string) * '=')
    print(stat_string)
    print(len(stat_string) * '=')
    plt.show()


def pie_chart(values,
              other_percent_thld='auto',
              title=None,
              figsize=(6, 6)):

    # Define the function generating pie chart labels
    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return '{p:.2f}%\n({v:d})'.format(p=pct, v=val)
        return my_autopct

    # Count number of occurences for each unique value
    if isinstance(values, dict):
        values_count = pd.DataFrame({'value': values.keys(), 'count': values.values()})
        values_n = sum(values_count['count'])
    else:
        values = pd.Series(values).rename('value').astype('str')
        values_count = values.value_counts().reset_index()
        values_count = values_count.rename(columns={'index': 'value', 'value': 'count'})
        values_n = len(values)

    unique_n = len(values_count)

    if other_percent_thld == 'auto':

        values_count = values_count.sort_values(by=['count'], ascending=False)

        pieces_fraction = 0
        small_pieces_n = 0
        tiny_pieces_n = 0
        for i, row in values_count.iterrows():
            piece_fraction = row['count'] / values_n
            pieces_fraction += piece_fraction

            if piece_fraction < 0.05:
                small_pieces_n += 1
            if piece_fraction < 0.01:
                tiny_pieces_n += 1
            if pieces_fraction > 0.9 or small_pieces_n > 2 or tiny_pieces_n > 0:
                other_percent_thld = piece_fraction * 100
                print('INFO: Automatically calculated other_percent_thld == {:.2f}.'
                      .format(other_percent_thld))
                break

    # Create mask to group rare values in the single 'Other' value based on other_percent_thld
    other_mask = (values_count['count'] / values_n) < (other_percent_thld * 0.01)
    other_n = other_mask.sum()

    if unique_n - other_n + 1 > 15:
        print('INFO: Consider increasing other_percent_thld parameter in order to'
              + ' include rare values in "Other" value.')

    if other_n > 1:
        # Group rare values
        values_count.loc[other_mask, 'value'] = 'Other'
        values_count = values_count.groupby('value').sum()
        other_fraction = values_count['count']['Other'] / values_n
        values_count = values_count.reset_index()

        if other_fraction > 0.5:
            print('INFO: Consider decreasing other_percent_thld parameter in order to'
                  + ' unwrap "Other" value.')
        print('INFO: {} pieces shown but "Other" also consists of {} distinct pieces.'
              .format(unique_n - other_n + 1, other_n))

    # Set colormap
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0.25, 0.85, len(values_count))]
    colors.reverse()

    # Sort values to match ordering of the color gradient
    values_count = values_count.sort_values(by=['count'])

    plt.style.use('default')
    plt.figure(figsize=figsize, dpi=80)
    plt.pie(values_count['count'], labels=values_count['value'], colors=colors,
            autopct=make_autopct(values_count['count']), startangle=180)
    plt.title(title)
    plt.show()
