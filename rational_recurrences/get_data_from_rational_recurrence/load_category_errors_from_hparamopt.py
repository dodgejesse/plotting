import sys
sys.path.append('/home/jessedd/projects/rational-recurrences/classification')
from experiment_params import ExperimentParams, get_categories
import load_learned_ngrams
import numpy as np
import os, re
import glob

np.set_printoptions(edgeitems=3,infstr='inf',
                    linewidth=9999, nanstr='nan', precision=5,
                    suppress=True, threshold=1000, formatter=None)


def get_data():
    learned_structures = {}
    worst = 0
    best = 1
    data = {}
    categories = get_categories()

    for category in categories:
        file_base = "/home/jessedd/projects/rational-recurrences/classification/logging/amazon_categories/" + category + "only_last_cs/hparam_opt/"
        file_name_endings = ["*none_*.txt", "*learned_*.txt", "*rho_entropy_*"]
        for file_name_ending in file_name_endings:
            file_names = glob.glob(file_base + file_name_ending)

            if len(file_names) != 0:
                for file_name in file_names:
                    cur_dev = load_from_file(file_name)
                    add_point_to_data(data, cur_dev, file_name, category)

                    if cur_dev < best:
                        best = cur_dev
                    if cur_dev > worst:
                        worst = cur_dev
    return data, categories, worst, best, learned_structures

def add_point_to_data(data, point, file_name, category):
    try:
        d_out = re.search('dout=(.+?)_',file_name).group(1)
        pattern = re.search('pattern=(.+?)_',file_name).group(1)
        sparsity = re.search('sparsity=(.+?)_',file_name).group(1)
    except AttributeError:
        print("PROBLEMS!")
        assert False

    # this is a bit of a hack
    if sparsity == "learned":
        pattern = "1-gram,2-gram,3-gram,4-gram"
    d_out = sum([int(x) for x in d_out.split(",")])
    if d_out not in data:
        data[d_out] = {}
    if pattern not in data[d_out]:
        data[d_out][pattern] = {}
    if sparsity not in data[d_out][pattern]:
        data[d_out][pattern][sparsity] = {}
    if category not in data[d_out][pattern][sparsity]:
        data[d_out][pattern][sparsity][category] = []
    data[d_out][pattern][sparsity][category].append(point)


def try_load_data(data, category, **kwargs):
    args = ExperimentParams(**kwargs)
    try:
        dev = load_from_file(args)
        data.append(dev)
        return args
    except FileNotFoundError:
        return


def num_training_examples():
    train_nums = {
        "original_mix/": 20001,
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
    
def load_from_file(path):
    lines = []
    with open(path, "r") as f:
        lines = f.readlines()

    if False:
        err = lines[-2]
        assert "best_valid" in err, str(args)
    else:
        err = lines[-1]
        assert "test_err" in err, str(args)
    
    return float(err.split(" ")[1])

if __name__ == "__main__":
    data, categories, worst, best, learned_structures = get_data()
    import pdb; pdb.set_trace()
    print(get_data.keys())
    sys.exit()
    
    data, worst, best, learned_structures = get_data(d_outs = ["24"],
             patterns = ["4-gram"],
             lrs = [0.001],
             sparsities = ["rho_entropy"],
             suffixes = ["_0","_1","_2","_3"])
    import pdb; pdb.set_trace()
    train_nums = num_training_examples()
    print(learned_structures)
    for d_out in ['24', '256']:
        sorted_data = []
        for category in learned_structures[d_out]:
            sorted_data.append([train_nums[category], learned_structures[d_out][category]])
        sorted_data.sort()
        for i in range(len(sorted_data)):
            print(sorted_data[i][1])
        print("")
        
