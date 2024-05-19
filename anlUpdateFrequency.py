import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def analyze_projects(data_file):
    # Read the processed data
    df = pd.read_excel(data_file, engine='openpyxl')

    # Convert 'pushed_at' to datetime for analysis
    df['pushed_at'] = pd.to_datetime(df['pushed_at']).dt.tz_localize(None)

    # Calculate update frequency (days since last push)
    today = datetime.now()
    df['days_since_last_push'] = (today - df['pushed_at']).dt.days

    # Calculate average update frequency and average stargazers_count for C and Go projects
    avg_update_freq_c = df[df['language'] == 'C']['days_since_last_push'].mean()
    avg_update_freq_go = df[df['language'] == 'Go']['days_since_last_push'].mean()

    avg_stars_c = df[df['language'] == 'C']['stargazers_count'].mean()
    avg_stars_go = df[df['language'] == 'Go']['stargazers_count'].mean()

    # Print conclusions
    conclusions = {
        'Average Update Frequency for C projects': f'{avg_update_freq_c:.2f} days',
        'Average Update Frequency for Go projects': f'{avg_update_freq_go:.2f} days',
        'Average Stars Count for C projects': f'{avg_stars_c:.2f}',
        'Average Stars Count for Go projects': f'{avg_stars_go:.2f}',
        'Comparison: C vs Go': f'C projects have {"higher" if avg_update_freq_c < avg_update_freq_go else "lower"} average update frequency and {"more" if avg_stars_c < avg_stars_go else "less"} average stars count than Go projects.'
    }

    for key, value in conclusions.items():
        print(f'{key}: {value}')

    # Plotting the results with improved styling
    languages = ['C', 'Go']
    avg_update_freq = [avg_update_freq_c, avg_update_freq_go]
    avg_stars = [avg_stars_c, avg_stars_go]

    # Plotting Average Update Frequency
    fig, ax1 = plt.subplots(figsize=(8, 6))

    color = 'tab:red'
    ax1.set_xlabel('Programming Language')
    ax1.set_ylabel('Average Update Frequency (Days Since Last Push)', color=color)
    ax1.bar(languages, avg_update_freq, color=color, alpha=0.7, edgecolor='black')
    ax1.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()

    # Save conclusions to a text file
    output_folder = 'output'
    os.makedirs(output_folder, exist_ok=True)

    with open(os.path.join(output_folder, 'conclusions.txt'), 'w') as f:
        for key, value in conclusions.items():
            f.write(f'{key}: {value}\n')

    # Save plot as image
    plt.savefig(os.path.join(output_folder, 'avg_update_frequency_plot.png'))

    plt.title('Average Update Frequency - C vs Go Language Projects Analysis')
    plt.show()


if __name__ == '__main__':
    data_file = 'github_projects_processed.xlsx'
    analyze_projects(data_file)
