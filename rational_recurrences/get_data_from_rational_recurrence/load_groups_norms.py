import sys
sys.path.append('/home/jessedd/projects/rational-recurrences/classification')
from experiment_params import ExperimentParams
import numpy as np

np.set_printoptions(edgeitems=3,infstr='inf',
                    linewidth=9999, nanstr='nan', precision=5,
                    suppress=True, threshold=1000, formatter=None)


def main():

    args = ExperimentParams(pattern="4-gram", d_out="24", depth = 1,
                            filename_prefix = "all_cs_and_equal_rho/",
                            dataset = "amazon_categories/original_mix/", use_rho = False, seed=None,
                            dropout= 0.4703, embed_dropout= 0.0805,rnn_dropout= 0.0027,
                            lr= 7.285E-02, weight_decay= 7.05E-06, clip_grad= 1.52, sparsity_type = "states",
                            reg_strength_multiple_of_loss = 1)

    import pdb; pdb.set_trace()

    vals = load_from_file(args)
    print(len(vals))



def from_file(args)

    norms = get_norms(args)


    
    threshold = 0.1
    #threshold = min(norms[-1][:,0])
    learned_ngrams = norms > threshold
    ngram_counts = [0] * (len(learned_ngrams[0]) + 1)
    weirdos = []

    for i in range(len(learned_ngrams)):
        cur_ngram = 0
        cur_weird = False
        for j in range(len(learned_ngrams[i])):
            if cur_ngram == j and learned_ngrams[i][j]:
                cur_ngram += 1
            elif cur_ngram == j and not learned_ngrams[i][j]:
                continue
            elif cur_ngram != j and learned_ngrams[i][j]:
                cur_weird = True
            elif cur_ngram != j and not learned_ngrams[i][j]:
                continue
        if cur_weird:
            weirdos.append(learned_ngrams[i])
        else:
            ngram_counts[cur_ngram] += 1
    
    print("0,1,2,3,4 grams: {}, num out of order: {}".format(str(ngram_counts), len(weirdos)))
    return ngram_counts

    

    
def get_norms(args):
    path = "/home/jessedd/projects/rational-recurrences/classification/logging/" + args.dataset
    path += args.file_name() + ".txt"


    lines = []
    with open(path, "r") as f:
        lines = f.readlines()

    if args.sparsity_type == "wfsa":
        vals = []
        for line in lines:
            try:
                vals.append(float(line))
            except:
                continue
    elif args.sparsity_type == "edges" or args.sparsity_type == "states":

        if args.sparsity_type == "edges":
            len_groups = 8
        else:
            len_groups = 4
        
        vals = []
        prev_line_was_data = False
        wfsas = []
        for line in lines:
            

            split_line = line.split(" ")
            if len(split_line) != len_groups*2 and prev_line_was_data:
                #print(line)
                prev_line_was_data = False
                vals.append(wfsas)
                wfsas = []
            else:
                edges = []
                for item in split_line:
                    try:
                        edges.append(float(item))
                    except:
                        continue
                #print(edges)
                #print(line)
                if len(edges) == len_groups:
                    prev_line_was_data = True
                    wfsas.append(edges)

        vals = vals[-1]
        vals = np.asarray(vals)


        assert vals.shape[0] == 24 # this is the number of WFSAs in the model
        assert vals.shape[1] == len_groups # this is the number of edges in each WFSA
        
            
                
    return vals


if __name__ == "__main__":
    main()
