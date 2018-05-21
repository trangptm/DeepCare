__author__ = 'phtra'

import sys

def filter_cutoff(inp_file, out_file):
    fi = open(inp_file, 'r')
    fo = open(out_file, 'w')

    for line in fi:
        ls = line.lower().split('\t')
        if ls[0] == 'atd_id':
            fo.write(line.lower())
            continue

        #ls[5]: ICD_code
        if (len(ls[5]) < 1) or (len(ls[5]) > 10):
            continue

        ls[5] = ls[5][:2]

        fo.write('\t'.join(ls))


if __name__ == '__main__':
    atd_file = 'preprocessed/' + sys.argv[1] + '/attendances.txt'
    out_file = 'preprocessed/' + sys.argv[1] + '/atd_filtered.txt'
    filter_cutoff(atd_file, out_file)