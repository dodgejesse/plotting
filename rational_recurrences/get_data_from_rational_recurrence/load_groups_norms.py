import sys
sys.path.append('/home/jessedd/projects/rational-recurrences/classification')
from experiment_params import ExperimentParams
import numpy as np

np.set_printoptions(edgeitems=3,infstr='inf',
                    linewidth=9999, nanstr='nan', precision=5,
                    suppress=True, threshold=1000, formatter=None)


def main():

    args = ExperimentParams()
    print(args)
    vals = load_from_file(args)
    #print(len(vals))
    #print(len(vals) % 256)
    import pdb; pdb.set_trace()
    print(len(vals))

    

    
def load_from_file(args):
    path = "/home/jessedd/projects/rational-recurrences/classification/logging/"
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
        vals = np.asarray(vals)
        assert vals.shape[1] == 36 # this is the number of WFSAs in the model
        assert vals.shape[2] == len_groups # this is the number of edges in each WFSA
        
            
                
    return vals


if __name__ == "__main__":
    main()
