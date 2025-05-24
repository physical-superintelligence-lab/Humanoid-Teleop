import glob
import json
import os
import re
import sys


def sanitize_filename(name):
    name = name.lower().replace(" ", "_")
    name = re.sub(r"[^\w\-]", "", name)
    return name


def process_json_files(directory="task_defs"):
    json_files = glob.glob(os.path.join(directory, "*.json"))

    if not json_files:
        print(f"No JSON files found in {directory}")
        return

    print(f"Found {len(json_files)} JSON file(s) to process")

    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            print(f"Processing file: {json_file}")

            task_category = os.path.basename(os.path.splitext(json_file)[0])

            for task in data["tasks"]:
                if "title" in task and "category" in task and "description" in task:
                    category_folder = sanitize_filename(task["category"])
                    title_folder = sanitize_filename(task["title"])

                    task_path = os.path.join(
                        "data", task_category, category_folder, title_folder
                    )
                    metadata_path = os.path.join(task_path, "metadata")

                    os.makedirs(metadata_path, exist_ok=True)

                    metadata = {
                        "title": task["title"],
                        "category": task["category"],
                        "description": task["description"],
                    }

                    with open(
                        os.path.join(metadata_path, "metadata.json"), "w"
                    ) as meta_file:
                        json.dump(metadata, meta_file, indent=2)

                    print(f"Created: {os.path.join(metadata_path, 'metadata.json')}")
                else:
                    print(f"Skipping task with missing fields: {task}")

        except json.JSONDecodeError:
            print(f"Error: {json_file} is not a valid JSON file")
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")


if __name__ == "__main__":

    directory = sys.argv[1] if len(sys.argv) > 1 else "task_defs"

    process_json_files(directory)
    print("Processing complete!")
