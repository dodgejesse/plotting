
import os



#categories = ['kitchen_&_housewares/', 'dvd/', 'books/', 'original_mix/']
categories = ['BERT']

def get_data(path):

    lines = []
    with open(path, "r") as f:
        lines = f.readlines()

    data = process_lines(lines)
    formatted_data = format_data(data)
    return {24:formatted_data}, categories, 0,0, None

def process_lines(lines):
    data = {}
    dataset_name = ""

    for line in lines:
        # lines we don't care about:
        if "baselines" in line or line.startswith("mean") or line.startswith("std") or line.startswith("n"):
            continue
        if line.startswith("Amazon"):
            dataset_name = line
            data[dataset_name] = []
        elif line == ",,,,,,,,,\n":
            dataset_name = ""
        else:
            data[dataset_name].append(line)
            
    return data

def format_data(data):
    f_data = init_formatted_data()

    for name in data:
        print(name)
        if not "BERT" in name:
            continue
        for i in range(5):
            learned_structures(data[name][i], data[name][i+5], f_data, name)
        #import pdb; pdb.set_trace()
    return f_data

def init_formatted_data():
    f_data = {
        '1-gram': {
            'none': {}
        },
        '2-gram': {
            'none': {}
        },
        '3-gram': {
            'none': {}
        },
        '4-gram': {
            'none': {}
        },
        '1-gram,2-gram,3-gram,4-gram': {
            'none': {},
            'l1-states-learned_goalparams=20': {},
            'l1-states-learned_goalparams=40': {},
            'l1-states-learned_goalparams=60': {},
            'l1-states-learned_goalparams=80': {}
        }
    }

    for ngram in f_data:
        for sparsity in f_data[ngram]:
            f_data[ngram][sparsity]['books/'] = []
            f_data[ngram][sparsity]['kitchen_&_housewares/'] = []
            f_data[ngram][sparsity]['dvd/'] = []
            f_data[ngram][sparsity]['original_mix/'] = []
            f_data[ngram][sparsity]['BERT'] = []

    return f_data
    
def learned_structures(states, errs, f_data, name):
    split_states = states.split(",")
    split_errs = errs.split(",")

    #if "BERT" in name:
    #import pdb; pdb.set_trace()
    
    for i in range(1,10):
        if i < 6:
            cur_state = get_state_from_index(i, name)
        else:
            cur_state = int(float(split_states[i].strip()))
        cur_err = 100*(1-float(split_errs[i].strip()))

        f_data[get_ngram_from_index(i)][get_sparsity_from_index(i)][name_map(name)].append([cur_err, cur_state])

def name_map(name):
    if "original" in name:
        return 'original_mix/'
    elif 'DVD' in name:
        return 'dvd/'
    elif 'housewares,' in name:
        return 'kitchen_&_housewares/'
    elif 'books' in name:
        return 'books/'
    elif 'BERT' in name:
        return 'BERT'
    else:
        assert False

def get_ngram_from_index(index):
    if index < 5:
        return "{}-gram".format(5-index)
    else:
        return '1-gram,2-gram,3-gram,4-gram'

def get_sparsity_from_index(index):
    if index < 6:
        return 'none'
    else:
        return 'l1-states-learned_goalparams={}'.format((5-(index - 5)) * 20)
    
def get_state_from_index(index, name):

    if "BERT" in name:
        if index < 5:
            return 12*(5-index)
        else:
            return 30

    else:
        if index < 5:
            return 24*(5-index)
        else:
            return 60
    
if __name__ == "__main__":
    data = get_data(path = "./data/data_test.csv")
    import pprint
    import pdb; pdb.set_trace()
    pprint.pprint(data)    
    print(data)
