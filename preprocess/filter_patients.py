__author__ = 'phtra'

from collections import OrderedDict
import sys

def is_header(line):
    if 'a' <= line[0] <= 'z': return True
    return False

def filter_patients(inp_file, out_file):
    fi = open(inp_file, 'r')
    fo = open(out_file, 'w')

    patnt_dict = OrderedDict()
    for line in fi:
        if is_header(line):
            fo.write(line)
            continue

        ls = line.split('\t')
        if ls[0] not in patnt_dict:
            patnt_dict[ls[0]] = ls[1]
            fo.write(line)
        else:
            if patnt_dict[ls[0]] != ls[1]:
                print line

if __name__ == '__main__':
    patnt_file = 'preprocessed/' + sys.argv[1] + '/patients.txt'
    out_file = 'preprocessed/' + sys.argv[1] + '/patnts_filtered.txt'
    filter_patients(patnt_file, out_file)