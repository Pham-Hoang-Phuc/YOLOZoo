# src/core/data_manager.py
import subprocess
import os

def check_and_pull_data(dvc_path):
    """
    Kiểm tra và pull dữ liệu từ DVC (Hỗ trợ cả Folder dataset và File weights)
    """
    if not dvc_path:
        return True

    # Xác định file/folder mục tiêu (bỏ đuôi .dvc)
    target_path = dvc_path.replace('.dvc', '')
    
    # Nếu file mục tiêu ĐÃ tồn tại, thì không làm gì cả
    if os.path.exists(target_path):
        # print(f"Found local file/dir: {target_path}")
        return True

    # Nếu file mục tiêu CHƯA có, nhưng file .dvc CÓ -> Pull về
    if os.path.exists(dvc_path):
        print(f"--> Local file {target_path} missing. Pulling from DVC...")
        try:
            subprocess.run(['dvc', 'pull', dvc_path], check=True)
            print(f"--> Pulled successfully: {target_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error pulling from DVC: {e}")
            return False
    else:
        print(f"Warning: Neither target '{target_path}' nor DVC file '{dvc_path}' found.")
        return False