import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def summarize_statistics(df):
    res = {}
    grouped = df.groupby('modus')

    res['mean_instances'] = grouped['num_instances'].mean()
    res['median_instances'] = grouped['num_instances'].median()
    res['mean_features'] = grouped['num_features'].mean()
    res['median_features'] = grouped['num_features'].median()
    res['mean_missing_values'] = grouped['num_missing_values'].mean()
    res['pct_with_missing'] = grouped.apply(lambda x: (x['num_missing_values'] > 0).mean() * 100) # percentage of datasets with missings
    # Compute proportions instead of counts
    res['pct_numeric_features'] = grouped.apply(lambda x: (x['num_numeric_features'] / x['num_features']).mean() * 100)
    res['pct_categorical_features'] = grouped.apply(lambda x: (x['num_categorical_features'] / x['num_features']).mean() * 100)

    return(pd.DataFrame(res).round(2).reset_index())

def correlation_heatmaps(df, group_col = 'modus'):
    groups = df[group_col].unique()
    num_groups = len(groups)

    fig, axes = plt.subplots(1, num_groups, figsize = (6 * num_groups, 6))

    for ax, group in zip(axes, groups):
        group_df = df[df[group_col] == group].select_dtypes(include=['number'])  # Nur numerische Spalten
        if not group_df.empty:
            sns.heatmap(group_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
            ax.set_title(f"Modus: {group}")

    plt.tight_layout()
    plt.show()

def scatter_plot(df, x_col, y_col,
                 x_label, y_label,
                 group_col = 'modus',
                 log_x = True, log_y = True):
    plt.figure(figsize = (10, 6))
    g = sns.scatterplot(data = df, x = x_col, y = y_col, hue = group_col, ci = 90, palette="husl")
    if log_x:
        g.set(xscale="log")
    if log_y:
        g.set(yscale="log")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def grouped_histogram(df, x_col, x_label, group_col = 'modus'):
    plt.figure(figsize = (10, 6))
    sns.histplot(data = df, x = x_col,
                 hue = group_col, palette = 'husl',
                 alpha = 0.6, multiple = 'dodge',
                 edgecolor = 'black', shrink = 0.8,
                 binwidth=25)
    plt.xlabel(x_label)
    plt.ylabel('Count')
    plt.show()

def grouped_boxplot(df, x_col, x_label, group_col = 'modus'):
    plt.figure(figsize = (4, 5))
    plt.rcParams.update({'font.size': 15})
    sns.boxplot(data = df, x = group_col, y = x_col, hue = group_col,
                palette = 'husl')
    plt.xlabel('Modus')
    plt.ylabel(x_label)