import os
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = "./data/tasks_4_11_g1/"
OUTPUT_DIR = "./zipped/tasks_4_11_g1/"

def zip_single_subfolder(task_folder, subfolder):
    subfolder_path = os.path.join(task_folder, subfolder)
    if not os.path.isdir(subfolder_path):
        return f"[Skipped] Not a folder: {subfolder_path}"

    task_name = os.path.basename(task_folder)
    output_task_dir = os.path.join(OUTPUT_DIR, task_name)
    os.makedirs(output_task_dir, exist_ok=True)

    zip_path = os.path.join(output_task_dir, f"{subfolder}.zip")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for foldername, _, filenames in os.walk(subfolder_path):
            for filename in filenames:
                filepath = os.path.join(foldername, filename)
                arcname = os.path.relpath(filepath, start=subfolder_path)
                zipf.write(filepath, arcname)

    return f"[Zipped] {zip_path}"

def zip_all_tasks_parallel(base_dir, output_dir, max_workers=6):
    tasks = []

    for task_type in os.listdir(base_dir):
        task_folder = os.path.join(base_dir, task_type)
        if not os.path.isdir(task_folder):
            continue

        for subfolder in os.listdir(task_folder):
            subfolder_path = os.path.join(task_folder, subfolder)
            if os.path.isdir(subfolder_path):
                tasks.append((task_folder, subfolder))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(zip_single_subfolder, task_folder, subfolder): (task_folder, subfolder)
            for task_folder, subfolder in tasks
        }

        for future in as_completed(futures):
            print(future.result())

if __name__ == "__main__":
    zip_all_tasks_parallel(BASE_DIR, OUTPUT_DIR, max_workers=8)
