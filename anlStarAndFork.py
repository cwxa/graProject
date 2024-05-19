import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats

def analyze_projects(data_file):
    # Read the processed data
    df = pd.read_excel(data_file, engine='openpyxl')

    # Convert 'pushed_at' to datetime for analysis
    df['pushed_at'] = pd.to_datetime(df['pushed_at']).dt.tz_localize(None)

    # Calculate update frequency (days since last push)
    today = datetime.now()
    df['days_since_last_push'] = (today - df['pushed_at']).dt.days

    # Print conclusions
    conclusions = {
        'Average Update Frequency for C projects': df[df['language'] == 'C']['days_since_last_push'].mean(),
        'Average Update Frequency for Go projects': df[df['language'] == 'Go']['days_since_last_push'].mean(),
        'Average Stars Count for C projects': df[df['language'] == 'C']['stargazers_count'].mean(),
        'Average Stars Count for Go projects': df[df['language'] == 'Go']['stargazers_count'].mean(),
        'Average Forks Count for C projects': df[df['language'] == 'C']['forks_count'].mean(),
        'Average Forks Count for Go projects': df[df['language'] == 'Go']['forks_count'].mean(),
    }

    for key, value in conclusions.items():
        print(f'{key}: {value:.2f}')

    # Perform t-tests for Stars and Forks
    stars_t_stat, stars_p_value = stats.ttest_ind(df[df['language'] == 'C']['stargazers_count'], df[df['language'] == 'Go']['stargazers_count'])
    forks_t_stat, forks_p_value = stats.ttest_ind(df[df['language'] == 'C']['forks_count'], df[df['language'] == 'Go']['forks_count'])

    # Print t-test results
    t_test_results = {
        'Stars T-Statistic': stars_t_stat,
        'Stars P-Value': stars_p_value,
        'Forks T-Statistic': forks_t_stat,
        'Forks P-Value': forks_p_value,
    }

    for key, value in t_test_results.items():
        print(f'{key}: {value:.4f}')

    # Plotting the results
    languages = ['C', 'Go']
    avg_stars = [df[df['language'] == lang]['stargazers_count'].mean() for lang in languages]
    avg_forks = [df[df['language'] == lang]['forks_count'].mean() for lang in languages]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plotting Boxplots for Stars
    stars_boxplot = df.boxplot(column='stargazers_count', by='language', ax=ax1)
    stars_boxplot.set_title(f'Stars Count by Language\nT-Stat: {stars_t_stat:.4f}, P-Value: {stars_p_value:.4f}')
    stars_boxplot.set_ylabel('Stars Count')

    # Plotting Boxplots for Forks
    forks_boxplot = df.boxplot(column='forks_count', by='language', ax=ax2)
    forks_boxplot.set_title(f'Forks Count by Language\nT-Stat: {forks_t_stat:.4f}, P-Value: {forks_p_value:.4f}')
    forks_boxplot.set_ylabel('Forks Count')

    plt.suptitle('C vs Go Language Projects Analysis')

    # Save conclusions and t-test results to a text file
    output_folder = 'output'
    os.makedirs(output_folder, exist_ok=True)

    with open(os.path.join(output_folder, 'conclusions_and_t_tests.txt'), 'w') as f:
        f.write('Conclusions:\n')
        for key, value in conclusions.items():
            f.write(f'{key}: {value:.2f}\n')

        f.write('\nT-Test Results:\n')
        for key, value in t_test_results.items():
            f.write(f'{key}: {value:.4f}\n')

    # Save boxplots as image
    plt.savefig(os.path.join(output_folder, 'github_projects_boxplots.png'))

    plt.show()

if __name__ == '__main__':
    data_file = 'github_projects_processed.xlsx'
    analyze_projects(data_file)
