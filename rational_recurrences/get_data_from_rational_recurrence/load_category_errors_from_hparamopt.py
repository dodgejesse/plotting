import sys
sys.path.append('/home/jessedd/projects/rational-recurrences/classification')
from experiment_params import ExperimentParams, get_categories
import load_learned_structure
import numpy as np
import os, re
import glob

np.set_printoptions(edgeitems=3,infstr='inf',
                    linewidth=9999, nanstr='nan', precision=5,
                    suppress=True, threshold=1000, formatter=None)


def get_data(count_num_params = False):
    learned_structures = {}
    worst = 0
    best = 1
    data = {}
    categories = ["kitchen_&_housewares/","dvd/", "books/", "original_mix/"] # get_categories()
    visited_files = []
    for base in ["", "structure_search/add_reg_term_to_loss/"]:
        for category in categories:
            file_base = "/home/jessedd/projects/rational-recurrences/classification/logging/amazon_categories/" + category
            #file_base += "only_last_cs/hparam_opt/" + base
            file_base += "all_cs_and_equal_rho/hparam_opt/" + base
            
            filename_endings = ["*none_*.txt", "*learned_*.txt", "*rho_entropy_*", "*regstrmultofloss=*_*", "*goalparams=*_*"]
            for filename_ending in filename_endings:
                #if "goalparams" in filename_ending:
                #    import pdb; pdb.set_trace()
                filenames = glob.glob(file_base + filename_ending)
                
                if len(filenames) != 0:
                    for filename in filenames:
                        if not valid_filename(filename, visited_files):
                            continue
                        #if "1-gram,2-gram,3-gram,4-gram_sparsity=l1-states-learned_goalparam" in filename and not "books" in filename:
                        #    import pdb; pdb.set_trace()
                        cur_dev = load_from_file(filename)
                        add_point_to_data(data, cur_dev, filename, category, count_num_params)

                        if cur_dev < best:
                            best = cur_dev
                        if cur_dev > worst:
                            worst = cur_dev
    return data, categories, worst, best, learned_structures

def valid_filename(filename, visited_files):
    if (filename.endswith(
        "_0.txt") or filename.endswith(
            "_1.txt") or filename.endswith(
                "_2.txt") or filename.endswith("_3.txt") or filename.endswith("_4.txt")) and filename not in visited_files:
        visited_files.append(filename)
        return True
    return False
        

def add_point_to_data(data, point, filename, category, count_num_params):
    try:
        d_out = re.search('dout=(.+?)_',filename).group(1)
        pattern = re.search('pattern=(.+?)_',filename).group(1)
        sparsity = re.search('sparsity=(.+?)_',filename).group(1)
    except AttributeError:
        print("PROBLEMS!")
        assert False

    if "goalparams" in filename:
        goalparams = re.search('goalparams=(.+?)_',filename).group(1)
        sparsity += "_goalparams=" + goalparams

    num_params = count_params(filename, d_out, pattern)
    # this is a bit of a hack
    #if sparsity == "learned" or sparsity == "l1-learned" or sparsity == "l1-states-learned":
    if "learned" in sparsity:
        d_out = "24"
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

    if count_num_params:
        data[d_out][pattern][sparsity][category].append([point, num_params])
    else:
        data[d_out][pattern][sparsity][category].append(point)

# note this will have problems!
def count_params(filename, d_out, pattern):
    if "rho_entropy" in filename.split("/")[-1] or "regstrmultofloss" in filename.split("/")[-1] or "states" in filename.split("/")[-1]:
        
        ngram_counts = [0,0,0,0]
        if "rho_entropy" in filename:
            pattern, d_out, frac_under_pointnine = load_learned_structure.entropy_rhos(filename, .9)
        elif "regstrmultofloss" in filename or "sparsity=states_" in filename:
            prox = "prox" in filename
            d_out, total_params = load_learned_structure.l1_group_norms(filename = filename, prox=prox)
            pattern = "1-gram,2-gram,3-gram,4-gram"
            

        split_pattern = pattern.split(",")
        split_d_out = d_out.split(",")
        
        assert len(split_pattern) == len(split_d_out)
        for i in range(len(split_pattern)):
            # to get the cur ngram num
            cur_ngram_num = int(split_pattern[i].split("-")[0]) - 1
            ngram_counts[cur_ngram_num] = int(split_d_out[i])
        split_pattern = ['1-gram', '2-gram', '3-gram', '4-gram']
        

    else:
        split_pattern = pattern.split(",")
        ngram_counts = d_out.split(",")

    assert len(split_pattern) == len(ngram_counts)
    num_params = 0
    for i in range(len(ngram_counts)):
        num_params = num_params + int(split_pattern[i].split("-")[0]) * int(ngram_counts[i])


    #if "learned" in filename:
    #    print(pattern, d_out, num_params)
    return num_params
    
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

    if True:
        err = lines[-2]
        assert "best_valid" in err, str(args)
    else:
        err = lines[-1]
        assert "test_err" in err, str(args)
    
    return float(err.split(" ")[1])

if __name__ == "__main__":
    data, categories, worst, best, learned_structures = get_data(True)

    print("")

    for pattern in data[24]:
        for sparsity in data[24][pattern]:
            for dataset in data[24][pattern][sparsity]:
                if len(data[24][pattern][sparsity][dataset]) > 5:
                    print(pattern, sparsity, dataset)
        
    import pdb; pdb.set_trace()
    print("")
    sys.exit()
