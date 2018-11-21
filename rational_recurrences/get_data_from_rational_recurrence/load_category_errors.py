import sys
sys.path.append('/home/jessedd/projects/rational-recurrences/classification')
from experiment_params import ExperimentParams, get_categories
import numpy as np

np.set_printoptions(edgeitems=3,infstr='inf',
                    linewidth=9999, nanstr='nan', precision=5,
                    suppress=True, threshold=1000, formatter=None)


def get_data():

    data = {}
    categories = get_categories()
    for d_out in ["24", "256", "6,6,6,6"]:
        if d_out not in data:
            data[d_out] = {}
            
        for pattern in ["4-gram", "3-gram", "2-gram", "1-gram", "4-gram,3-gram,2-gram,1-gram"]:
            if pattern not in data[d_out]:
                data[d_out][pattern] = {}
            for category in categories:
                
                args = ExperimentParams(pattern=pattern, d_out = d_out, depth = 1, filename_prefix="only_last_cs/", use_last_cs=True, lr=0.00025, dataset = "amazon_categories/" + category)

                
                #print(args)
                try:
                    dev = load_from_file(args)
                    data[d_out][pattern][category] = dev
                except:
                    continue
                #print(len(vals))
                #print(len(vals) % 256)
                #import pdb; pdb.set_trace()
                #print(len(vals))

    return data
    
def num_training_examples():
    train_nums = {
        "apparel/": 1082,
        "automotive/": 122,
        "baby/": 720,
        "beauty/": 394,
        "books/": 20000,
        "camera_&_photo/": 880,
        "cell_phones_&_service/": 308,
        "computer_&_video_games/": 368,
        "dvd/": 14066,
        "electronics/": 4040,
        "gourmet_food/": 168,
        "grocery/": 282,
        "health_&_personal_care/": 1208,
        "jewelry_&_watches/": 234,
        "kitchen_&_housewares/": 3298,
        "magazines/": 776,
        "musical_instruments/": 40,
        "music/": 11818,
        "office_products/": 52,
        "outdoor_living/": 264,
        "software/": 734,
        "sports_&_outdoors/": 858,
        "tools_&_hardware/": 12,
        "toys_&_games/": 2056,
        "video/": 4512,
        }
    return train_nums
    
def load_from_file(args):
    path = "/home/jessedd/projects/rational-recurrences/classification/logging/" + args.dataset
    path += args.file_name() + ".txt"

    lines = []
    with open(path, "r") as f:
        lines = f.readlines()

    for line in lines:
        if "best_valid" in line:
            return float(line.split(" ")[1])
                


if __name__ == "__main__":
    total = 0
    data = get_data()
    for d_out in data:
        for pattern in data[d_out]:
            for category in data[d_out][pattern]:
                total += 1
                print(d_out, pattern, category, data[d_out][pattern][category])
    print(total)
    #num_training_examples()
