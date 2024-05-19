import pandas as pd

def preprocess_data(input_file, output_file, c_projects_count=100, go_projects_count=100):
    # Read Excel file
    df = pd.read_excel(input_file, engine='openpyxl')

    # Filter C and Go language projects
    c_projects = df[df['language'] == 'C'].head(c_projects_count)
    go_projects = df[df['language'] == 'Go'].head(go_projects_count)

    # Concatenate the selected projects
    selected_projects = pd.concat([c_projects, go_projects])

    # Columns to drop, including 'watchers_count'
    columns_to_drop = [
        'html_url', 'url', 'forks_url', 'keys_url', 'collaborators_url', 'teams_url',
        'hooks_url', 'issue_events_url', 'events_url', 'assignees_url', 'branches_url',
        'tags_url', 'blobs_url', 'git_tags_url', 'git_refs_url', 'trees_url', 'statuses_url',
        'languages_url', 'stargazers_url', 'contributors_url', 'subscribers_url',
        'subscription_url', 'commits_url', 'git_commits_url', 'comments_url',
        'issue_comment_url', 'contents_url', 'compare_url', 'merges_url', 'archive_url',
        'downloads_url', 'issues_url', 'pulls_url', 'milestones_url',
        'notifications_url', 'labels_url', 'releases_url', 'deployments_url',
        'git_url', 'ssh_url', 'clone_url', 'svn_url', 'homepage',
        'id', 'node_id', 'mirror_url', 'archived', 'disabled', 'license', 'allow_forking',
        'web_commit_signoff_required', 'topics', 'permissions', 'score',
        'watchers_count'  # Adding 'watchers_count' to be dropped
    ]

    # Drop specified columns
    selected_projects = selected_projects.drop(columns=columns_to_drop)
    print(selected_projects.columns)

    # Save processed data to a new Excel file
    selected_projects.to_excel(output_file, index=False)

if __name__ == '__main__':
    input_file = 'github_projects.xlsx'
    output_file = 'github_projects_processed.xlsx'

    preprocess_data(input_file, output_file, c_projects_count=100, go_projects_count=100)
