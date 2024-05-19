import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def analyze_boolean_features(data_file):
    # Read the processed data
    df = pd.read_excel(data_file, engine='openpyxl')

    # Filter relevant columns and rows for C and Go languages
    relevant_columns = ['language', 'has_issues', 'has_projects', 'has_downloads', 'has_wiki', 'has_pages',
                        'has_discussions']
    filtered_df = df[relevant_columns][(df['language'] == 'C') | (df['language'] == 'Go')]

    # Count the occurrences of TRUE and FALSE for each boolean feature
    count_df = filtered_df.groupby('language').sum()

    # Calculate the proportions of TRUE for each boolean feature
    proportions_df = count_df / filtered_df.groupby('language').size().values[:, None]

    # Print conclusions
    conclusions = {}
    for feature in proportions_df.columns:
        c_proportion = proportions_df.loc['C', feature]
        go_proportion = proportions_df.loc['Go', feature]
        conclusions[f'Proportion of {feature} in C projects'] = '{:.2%}'.format(c_proportion)
        conclusions[f'Proportion of {feature} in Go projects'] = '{:.2%}'.format(go_proportion)

        # Modify the comparison statement
        comparison_result = 'C projects have {} proportion of {} than Go projects.'.format(
            "higher" if c_proportion > go_proportion else "lower", feature
        )
        conclusions[f'Comparison: C vs Go - {feature}'] = comparison_result

    for key, value in conclusions.items():
        print(f'{key}: {value}')

    # Plotting the results with improved styling
    features = proportions_df.columns
    c_proportions = proportions_df.loc['C'].values
    go_proportions = proportions_df.loc['Go'].values

    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.35
    index = range(len(features))
    ax.bar(index, c_proportions, bar_width, label='C', alpha=0.7, edgecolor='black')
    ax.bar([i + bar_width for i in index], go_proportions, bar_width, label='Go', alpha=0.7, edgecolor='black')

    ax.set_xlabel('Boolean Features')
    ax.set_ylabel('Proportion')
    ax.set_title('Proportion of Boolean Features in C and Go Language Projects')
    ax.set_xticks([i + bar_width / 2 for i in index])
    ax.set_xticklabels(features)
    ax.legend()

    plt.tight_layout()

    # Save conclusions to a text file
    output_folder = 'output'
    os.makedirs(output_folder, exist_ok=True)

    with open(os.path.join(output_folder, 'conclusions.txt'), 'w') as f:
        for key, value in conclusions.items():
            f.write(f'{key}: {value}\n')

    # Save plot as image
    plt.savefig(os.path.join(output_folder, 'boolean_features_plot.png'))

    # Display the plot
    plt.show()


if __name__ == '__main__':
    data_file = 'github_projects_processed.xlsx'
    analyze_boolean_features(data_file)
