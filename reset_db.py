# reset_db.py
import os
import shutil
import time
from config import Config

def reset_database():
    # 1. 创建备份文件夹
    backup_root = "01_RAG\backups"
    os.makedirs(backup_root, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(backup_root, f"backup_{timestamp}")
    os.makedirs(backup_path, exist_ok=True)

    print(f"🚀 开始备份并清理环境...")

    # 2. 备份并移除旧数据
    targets = [Config.DB_DIR, Config.PROCESSED_RECORD_FILE]
    
    for target in targets:
        if os.path.exists(target):
            # 备份
            dest = os.path.join(backup_path, os.path.basename(target))
            if os.path.isdir(target):
                shutil.copytree(target, dest)
                shutil.rmtree(target) # 清理
            else:
                shutil.copy2(target, dest)
                os.remove(target) # 清理
            print(f"✅ 已处理: {target} -> 备份至 {backup_path}")
        else:
            print(f"ℹ️ 未找到目标，跳过: {target}")

    print(f"\n✨ 清理完毕！现在你可以重新运行 batch_ingest.py 了。")

if __name__ == "__main__":
    reset_database()
