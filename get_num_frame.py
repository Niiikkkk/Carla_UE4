import os

tot_frames = 0
root = os.listdir("output")
for folder in root:
    path = os.path.join("output", folder, "rgb")
    n = len(os.listdir(path))
    tot_frames += n
print(tot_frames)