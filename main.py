import os
import requests
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor
from zipfile import ZipFile
from ratelimit import limits, sleep_and_retry
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




CONFIG = {
    'github_token': '',#'github_token': '',请申请后填入此处

    'languages': ['go', 'c'],
    'pages': range(1, 8),
    'excel_filename': 'github_projects.xlsx',
    'code_dir': 'code',
}

# 一级速率限制：每分钟请求次数
FIRST_RATELIMIT_LIMIT = 5000
FIRST_RATELIMIT_PERIOD = 60

# 二级速率限制：每秒请求次数
SECOND_RATELIMIT_LIMIT = 30
SECOND_RATELIMIT_PERIOD = 1

# 速率限制装饰器
@sleep_and_retry
@limits(calls=SECOND_RATELIMIT_LIMIT, period=SECOND_RATELIMIT_PERIOD)
def api_request(url, headers):
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response

def handle_rate_limit(response, owner, repo):
    reset_time = int(response.headers.get('x-ratelimit-reset', 0))
    remaining_requests = int(response.headers.get('x-ratelimit-remaining', 1))

    if remaining_requests == 0:
        sleep_time = max(0, reset_time - int(time.time()) + 1)
        logger.info(f"Rate limit reached for {owner}/{repo}. Waiting for {sleep_time} seconds before retrying...")
        time.sleep(sleep_time)
    else:
        logger.info(f"Rate limit reached for {owner}/{repo}. Waiting for a minute before retrying...")
        time.sleep(60)

def get_github_projects(language, per_page=100, page=1, token=None):
    url = f'https://api.github.com/search/repositories?q=language:{language}&sort=stars&order=desc&page={page}&per_page={per_page}'
    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/json',
    }
    response = api_request(url, headers=headers)

    projects = response.json().get('items', [])
    return projects

def get_repository_contents(owner, repo, token=None):
    url = f'https://api.github.com/repos/{owner}/{repo}/contents'
    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/json',
    }
    response = api_request(url, headers=headers)

    contents = response.json()
    return contents

def save_to_excel(projects, filename):
    df = pd.DataFrame(projects)
    df.to_excel(filename, index=False)

def download_and_save_code_zip(owner, repo, token, code_dir, language):
    logger.info(f"Downloading {owner}/{repo} source code as a zip file")

    url = f'https://api.github.com/repos/{owner}/{repo}/zipball/master'
    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/json',
    }

    try:
        response = api_request(url, headers=headers)
    except requests.HTTPError as e:
        if e.response.status_code == 404:
            logger.error(f"Error processing {owner}/{repo}: {e}")
            return
        else:
            raise

    zip_content = response.content

    # 构建zip文件路径
    zip_file_path = os.path.join(code_dir, language, f'{owner}_{repo}_source_code.zip')

    # 保存zip文件
    with open(zip_file_path, 'wb') as zip_file:
        zip_file.write(zip_content)

    # 解压缩zip文件到相应目录
    extract_dir = os.path.join(code_dir, language, f'{owner}_{repo}_code')
    with ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    # 清理掉下载的zip文件
    os.remove(zip_file_path)

def process_project(project, token, code_dir, existing_projects):
    owner = project['owner']['login']
    repo = project['name']
    language = project.get('language', 'unknown')  # 如果没有语言信息，默认为'unknown'

    # 检查是否已经处理过该项目
    if f'https://github.com/{owner}/{repo}' in existing_projects:
        logger.info(f"Skipping already processed project: {owner}/{repo}")
        return

    repo_dir = os.path.join(code_dir, language, f'{owner}_{repo}_code')
    os.makedirs(repo_dir, exist_ok=True)

    try:
        contents = get_repository_contents(owner, repo, token=token)

        if not isinstance(contents, list):
            logger.warning(f"Skipping invalid contents for {owner}/{repo}")
            return

        for content in contents:
            if isinstance(content, dict) and content.get('type') == 'file':
                download_and_save_code_zip(owner, repo, token, code_dir, language)
                break
            else:
                logger.warning(f"Skipping non-file content: {content.get('path')}")

    except Exception as e:
        logger.error(f"Error processing {owner}/{repo}: {e}")

def get_all_projects(language, total_projects, page, token):
    projects = get_github_projects(language, page=page, token=token)
    if not projects:
        return None

    total_projects += len(projects)
    return projects, total_projects

def main():
    all_projects = []
    existing_projects = set()

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(get_all_projects, language, 0, page, CONFIG['github_token']) for language in CONFIG['languages'] for page in CONFIG['pages']]

        for future in futures:
            result = future.result()
            if result:
                projects, total_projects = result
                all_projects.extend(projects)

    save_to_excel(all_projects, CONFIG['excel_filename'])

    os.makedirs(CONFIG['code_dir'], exist_ok=True)

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_project, project, CONFIG['github_token'], CONFIG['code_dir'], existing_projects) for project in all_projects]

        for future in futures:
            future.result()

if __name__ == '__main__':
    main()
