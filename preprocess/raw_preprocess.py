__author__ = 'phtra'

import re
import sys

raw_data_path = 'list_raw_data.txt'

def preprocess(in_file, out_file):
    fi = open(in_file, 'r')
    fo = open(out_file, 'w')
    for line in fi:
        sentence = re.sub(r'[|]', '\t', line)
        fo.write(sentence)

def main(raw_path = raw_data_path,
         disease = '' # disease: diabetes, mental
         ):
    f_raw = open(raw_path, 'r')

    for line in f_raw:
        filename = line.split()[0]
        inp_file = 'raw/' + disease + '/' + filename
        out_file = 'preprocessed/' + disease + '/' + filename
        preprocess(inp_file, out_file)

if __name__ == '__main__':
    main(disease=sys.argv[1])