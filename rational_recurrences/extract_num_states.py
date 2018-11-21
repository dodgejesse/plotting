import sys
sys.path.append('/home/jessedd/projects/rational-recurrences/classification')
from experiment_params import ExperimentParams
import get_data_from_rational_recurrence.load_groups_norms as load_norms

import numpy as np

def main():

    args = ExperimentParams()
    norms = load_norms.load_from_file(args)


    
    threshold = 0.002
    #threshold = min(norms[-1][:,0])
    learned_ngrams = norms[-1] > threshold
    ngram_counts = [0] * (len(learned_ngrams[0]) + 1)
    weirdos = []
    import pdb; pdb.set_trace()
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
    print("")
    print(weirdos)



if __name__ == "__main__":
    main()

