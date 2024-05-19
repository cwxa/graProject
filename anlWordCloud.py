import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from wordcloud import WordCloud
from collections import Counter


def generate_word_cloud(text, language, output_folder):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {language} Projects Descriptions')

    # Save word cloud as image
    plt.savefig(os.path.join(output_folder, f'wordcloud_{language}.png'))
    plt.show()


def analyze_projects(data_file):
    # Read the processed data
    df = pd.read_excel(data_file, engine='openpyxl')

    # Generate word cloud for C projects
    c_projects = df[df['language'] == 'C']['description'].dropna()
    c_text = ' '.join(c_projects)
    generate_word_cloud(c_text, 'C', 'output')

    # Generate word cloud for Go projects
    go_projects = df[df['language'] == 'Go']['description'].dropna()
    go_text = ' '.join(go_projects)
    generate_word_cloud(go_text, 'Go', 'output')


if __name__ == '__main__':
    data_file = 'github_projects_processed.xlsx'
    analyze_projects(data_file)
