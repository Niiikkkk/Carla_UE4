from sklearn.model_selection import train_test_split
import os
import json
import re
import random

ROOT = "/normal_dataset/"  # Root directory of the dataset

def extact_frame_number(filename):
    """Extract an integer from the filename for comparison."""
    nums = re.findall(r'\d+', filename)
    return int(nums[-1]) if nums else None

def sorted_files(path):
    print(path)
    return sorted(os.listdir(path), key=lambda x: extact_frame_number(x) if extact_frame_number(x) is not None else -1)

def get_dict(array):
    return [{
        "id": folder,
        "rgb": rgb_path,
        "semantic_rgb": rgb_path.replace("/rgb/", "/semantic/original/"),
    } for rgb_path in array]

samples = []
train = []
val = []

# Iterate over numbered folders (1, 2, 3, ...)
for folder in sorted(os.listdir(ROOT)):
    sample_dir = os.path.join(ROOT, folder)
    if not os.path.isdir(sample_dir):
        continue

    rgb_path = os.path.join(sample_dir, "rgb")
    lidar_path = os.path.join(sample_dir, "lidar")
    sem_rgb_path = os.path.join(sample_dir, "semantic/original")
    sem_lidar_path = os.path.join(sample_dir, "semantic_lidar")

    rgb = sorted_files(rgb_path)
    # lidar = sorted_files(lidar_path)
    # lidar.remove("raw")
    sem_rgb = sorted_files(sem_rgb_path)
    # sem_lidar = sorted_files(sem_lidar_path)
    # sem_lidar.remove("raw")

    #Check FILE CONSISTENCY

    print(len(rgb), len(sem_rgb))

    n = len(rgb)
    assert n == len(sem_rgb), f"Mismatch in number of frames in folder {folder}"

    path = os.path.join(ROOT, folder)
    train_split = [os.path.join(path, "rgb", f) for f in random.sample(rgb, int(0.8 * n))]
    val_split = [os.path.join(path, "rgb", f) for f in rgb if os.path.join(path, "rgb", f) not in train_split]

    train.extend(get_dict(train_split))
    val.extend(get_dict(val_split))

# Save full dataset JSON
# with open("dataset.json", "w") as f:
#     json.dump({"samples": samples}, f, indent=4)


# Create train/validation split (80/20)
# train_samples, val_samples = train_test_split(samples, test_size=0.2, shuffle=True, random_state=42)

with open("train_new.json", "w") as f:
    json.dump({"samples": train}, f, indent=4)

with open("val_new.json", "w") as f:
    json.dump({"samples": val}, f, indent=4)

print("Completed: train.json, val.json")

