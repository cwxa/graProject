import os
import shutil
import logging
from tqdm import tqdm

# 配置日志记录
logging.basicConfig(filename='move_c_files.log', level=logging.INFO, format='%(asctime)s %(message)s')

def move_c_files_to_single_directory(src_base_path, dest_base_path):
    # 确保源路径存在
    if not os.path.exists(src_base_path):
        logging.error(f"The source path {src_base_path} does not exist.")
        return

    # 确保目标路径存在，如果不存在则创建
    os.makedirs(dest_base_path, exist_ok=True)

    # 记录移动的文件数量
    moved_files_count = 0

    # 获取所有.c文件的路径
    c_files = []
    for root, dirs, files in os.walk(src_base_path):
        for file in files:
            if file.endswith('.c'):
                c_files.append(os.path.join(root, file))

    # 使用tqdm创建进度条
    with tqdm(total=len(c_files), desc='Moving .c files', unit='file') as pbar:
        # 移动.c文件到目标文件夹
        for src_file_path in c_files:
            try:
                # 构建目标路径，直接使用目标基路径，不使用相对路径
                dest_file_path = os.path.join(dest_base_path, os.path.basename(src_file_path))

                # 确认目标文件不存在，或者提示用户是否覆盖
                if os.path.exists(dest_file_path):
                    logging.warning(f"File {dest_file_path} already exists. Skipping.")
                    pbar.update(1)  # 更新进度条，但不计数为移动成功的文件
                    continue

                # 移动文件
                shutil.move(src_file_path, dest_file_path)
                moved_files_count += 1
                logging.info(f"Moved: {src_file_path} to {dest_file_path}")
                pbar.update(1)  # 更新进度条
            except Exception as e:
                logging.error(f"Failed to move {src_file_path} : {e}")

    logging.info(f"Total number of .c files moved: {moved_files_count}")

# 替换下面的路径为你的C项目文件夹路径
c_projects_path = 'E:\graduation project\code\C'
# 替换下面的路径为你想要保存.c文件的目标文件夹路径
c_files_target_path = 'E:\graduation project\csrc'

# 调用函数
move_c_files_to_single_directory(c_projects_path, c_files_target_path)