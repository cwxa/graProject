import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def analyze_collaboration_indicators(data_file):
    # Read the processed data
    df = pd.read_excel(data_file, engine='openpyxl')

    # Filter relevant columns and rows for C and Go languages
    relevant_columns = ['language', 'open_issues_count', 'has_discussions']
    filtered_df = df[relevant_columns][(df['language'] == 'C') | (df['language'] == 'Go')]

    # Convert 'has_discussions' to 0 or 1
    df['has_discussions'] = df['has_discussions'].astype(int)

    # Calculate average open issues count and discussion presence
    avg_open_issues_count_c = df[df['language'] == 'C']['open_issues_count'].mean()
    avg_open_issues_count_go = df[df['language'] == 'Go']['open_issues_count'].mean()

    avg_discussions_c = df[df['language'] == 'C']['has_discussions'].mean()
    avg_discussions_go = df[df['language'] == 'Go']['has_discussions'].mean()

    # Perform t-tests for open issues count and discussions
    open_issues_t_stat, open_issues_p_value = stats.ttest_ind(
        df[df['language'] == 'C']['open_issues_count'], df[df['language'] == 'Go']['open_issues_count']
    )

    discussions_t_stat, discussions_p_value = stats.ttest_ind(
        df[df['language'] == 'C']['has_discussions'], df[df['language'] == 'Go']['has_discussions']
    )

    # Print t-test results
    t_test_results = {
        'Open Issues T-Statistic': open_issues_t_stat,
        'Open Issues P-Value': open_issues_p_value,
        'Discussions T-Statistic': discussions_t_stat,
        'Discussions P-Value': discussions_p_value,
    }

    for key, value in t_test_results.items():
        print(f'{key}: {value:.4f}')

    # Define colors for discussions bar chart
    colors_discussions = ['lightblue', 'lightgreen']

    # Plotting the results with improved styling
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Boxplot for open issues count
    boxplot_open_issues = df.boxplot(column='open_issues_count', by='language', ax=axes[0], patch_artist=True)
    colors_open_issues = ['lightblue', 'lightgreen']

    for patch, color in zip(boxplot_open_issues.get_children(), colors_open_issues):
        if isinstance(patch, plt.Artist) and isinstance(patch, plt.Polygon):
            patch.set_facecolor(color)

    axes[0].set_xlabel('Programming Language')
    axes[0].set_ylabel('Open Issues Count')
    axes[0].set_title(f'Boxplot of Open Issues Count - C vs Go\nT-Stat: {open_issues_t_stat:.4f}, P-Value: {open_issues_p_value:.4f}')

    # Bar chart for discussions presence
    bar_chart_discussions = df.groupby('language')['has_discussions'].mean().plot(kind='bar', ax=axes[1], color=colors_discussions)
    axes[1].set_xlabel('Programming Language')
    axes[1].set_ylabel('Average Discussions Presence (0: No, 1: Yes)')
    axes[1].set_title(f'Bar Chart of Discussions Presence - C vs Go\nT-Stat: {discussions_t_stat:.4f}, P-Value: {discussions_p_value:.4f}')
    axes[1].set_xticklabels(['C', 'Go'], rotation=0)

    # Adjust layout for better spacing
    fig.tight_layout()

    # Save plot as image
    output_folder = 'output'
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, 'collaboration_indicators_plots.png'))

    # Display the plots
    plt.show()

if __name__ == '__main__':
    data_file = 'github_projects_processed.xlsx'
    analyze_collaboration_indicators(data_file)
