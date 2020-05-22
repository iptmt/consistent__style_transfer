import os
import random

def read(file):
    with open(file, 'r') as f:
        return [line.strip() for line in f]

def write(lines, file):
    with open(file, 'w+') as f:
        for line in lines:
            f.write(line + "\n")


topic = "yelp"
out_dir = "~/"
in_dir = f"../../data/{topic}/"

file_names = os.listdir(in_dir)

# transform
for name in file_names:
    if "reference" in name:
        continue
    lines = read(in_dir + name)
    lines = [name.split(".")[-1] + " " + l for l in lines]
    write(lines, out_dir + name)

# merge train
lines_0 = ["0" + " " + l for l in read(in_dir + "style.train.0")]
lines_1 = ["1" + " " + l for l in read(in_dir + "style.train.1")]
lines_train = lines_0 + lines_1
random.shuffle(lines_train)
write(lines_train, out_dir + "train.txt")

# merge dev
lines_0 = ["0" + " " + l for l in read(in_dir + "style.dev.0")]
lines_1 = ["1" + " " + l for l in read(in_dir + "style.dev.1")]
lines_dev = lines_0 + lines_1
random.shuffle(lines_dev)
write(lines_dev, out_dir + "dev.txt")