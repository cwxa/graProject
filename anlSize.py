import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def compare_project_sizes(data_file):
    # Read the processed data
    df = pd.read_excel(data_file, engine='openpyxl')

    # Filter relevant columns and rows for C and Go languages
    relevant_columns = ['language', 'size']
    filtered_df = df[relevant_columns][(df['language'] == 'C') | (df['language'] == 'Go')]

    # Calculate average project size for C and Go
    avg_size_c = df[df['language'] == 'C']['size'].mean()
    avg_size_go = df[df['language'] == 'Go']['size'].mean()

    # Perform t-test for project sizes
    size_t_stat, size_p_value = stats.ttest_ind(
        df[df['language'] == 'C']['size'], df[df['language'] == 'Go']['size']
    )

    # Print t-test results
    t_test_results = {
        'Project Size T-Statistic': size_t_stat,
        'Project Size P-Value': size_p_value,
    }

    for key, value in t_test_results.items():
        print(f'{key}: {value:.4f}')

    # Plotting the results with improved styling
    fig, ax = plt.subplots(figsize=(8, 6))

    # Bar plot for average project sizes
    languages = ['C', 'Go']
    avg_sizes = [avg_size_c, avg_size_go]
    ax.bar(languages, avg_sizes, color=['lightblue', 'lightgreen'], alpha=0.7, edgecolor='black')

    ax.set_xlabel('Programming Language')
    ax.set_ylabel('Average Project Size')
    ax.set_title(f'Bar Plot of Average Project Sizes - C vs Go\nT-Stat: {size_t_stat:.4f}, P-Value: {size_p_value:.4f}')

    # Save plot as image
    output_folder = 'output'
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, 'average_project_sizes_barplot.png'))

    # Display the plot
    plt.show()

if __name__ == '__main__':
    data_file = 'github_projects_processed.xlsx'
    compare_project_sizes(data_file)
