import sys
sys.path.append('/home/jessedd/projects/rational-recurrences/classification')
from experiment_params import ExperimentParams, get_categories
import load_learned_ngrams
import numpy as np

np.set_printoptions(edgeitems=3,infstr='inf',
                    linewidth=9999, nanstr='nan', precision=5,
                    suppress=True, threshold=1000, formatter=None)


def get_data():
    worst = 0
    best = 1
    data = {}
    categories = get_categories()
    for d_out in ["24", "256", "6,6,6,6", "64,64,64,64"]:
        if d_out not in data:
            data[d_out] = {}
            
        for pattern in ["4-gram", "3-gram", "2-gram", "1-gram", "1-gram,2-gram,3-gram,4-gram"]:
            if pattern not in data[d_out]:
                data[d_out][pattern] = {}


            for lr in [0.00025, 0.001]:
                if lr not in data[d_out][pattern]:
                    data[d_out][pattern][lr] = {}

                for sparsity in ["rho_entropy", "none"]:
                    if sparsity not in data[d_out][pattern][lr]:
                        data[d_out][pattern][lr][sparsity] = {}

                    for category in categories:

                        if sparsity == "rho_entropy":
                            args = try_load_data(data[d_out][pattern][lr][sparsity], category,
                                          pattern=pattern, d_out = d_out, depth = 1, filename_prefix="only_last_cs/",
                                          use_last_cs=True, lr=lr, dataset = "amazon_categories/" + category,
                                          sparsity_type=sparsity, reg_strength=0.01)
                            if not args:
                                continue
                            

                            file_base = "/home/jessedd/projects/rational-recurrences/classification/logging/amazon_categories/" + category
                            learned_pattern, learned_d_out, frac_under_pointnine = load_learned_ngrams.from_file(
                                file_base + args.file_name() + ".txt")
                            if "belowpointnine" not in data[d_out][pattern][lr]:
                                data[d_out][pattern][lr]["belowpointnine"] = {}
                            data[d_out][pattern][lr]["belowpointnine"][category] = round(frac_under_pointnine, 2)
                            

                            if "learned" not in data[d_out][pattern][lr]:
                                data[d_out][pattern][lr]["learned"] = {}
                            
                            try_load_data(data[d_out][pattern][lr]["learned"], category,
                                          pattern = learned_pattern, d_out=learned_d_out, lr=lr, filename_prefix="only_last_cs/",
                                          dataset = "amazon_categories/" + category, use_last_cs=True, learned_structure=True,
                                          use_rho = False)

                        else:

                            try_load_data(data[d_out][pattern][lr][sparsity], category,
                                          pattern=pattern, d_out = d_out, depth = 1, filename_prefix="only_last_cs/",
                                          use_last_cs=True, lr=lr, dataset = "amazon_categories/" + category,
                                          sparsity_type=sparsity, reg_strength=0)


                        if category in data[d_out][pattern][lr][sparsity]:
                            if data[d_out][pattern][lr][sparsity][category] > worst:
                                worst = data[d_out][pattern][lr][sparsity][category]
                            if data[d_out][pattern][lr][sparsity][category] < best:
                                best = data[d_out][pattern][lr][sparsity][category]
                            
    return data, worst, best

def try_load_data(data, category, **kwargs):
    args = ExperimentParams(**kwargs)
    try:
        dev = load_from_file(args)
        data[category] = dev
        return args
    except FileNotFoundError:
        return


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

    best_valid_line = lines[-2]
    assert "best_valid" in best_valid_line, str(args)
    
    return float(best_valid_line.split(" ")[1])
                


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
