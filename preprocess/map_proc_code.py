__author__ = 'phtra'

from collections import OrderedDict
import sys

def build_dict(file_code):
    fi = open(file_code, 'r')

    dictionary = OrderedDict()
    for line in fi:
        ls = line.lower().split('\t')
        if ls[0] == 'type': continue

        dictionary[ls[1]] = ls[15]

    return dictionary

def mapping(file_proc, file_code, file_out):
    f_proc = open(file_proc, 'r')
    f_out = open(file_out, 'w')

    dictionary = build_dict(file_code)

    for line in f_proc:
        ls = line.lower().split('\t')
        if ls[0] == 'dgpro_refno' or ls[3] != 'proce':
            f_out.write(line.lower())
            continue

        if ls[6] in dictionary:
            ls[6] = dictionary[ls[6]]
        else: ls[6] = '0'

        f_out.write('\t'.join(ls))

if __name__ == '__main__':
    file_proc = 'preprocessed/' + sys.argv[1] + '/diagnosis_procedures.txt'
    file_code = 'CODE/ICD-10-AM-12-13-procedure.txt'
    file_out = 'preprocessed/' + sys.argv[1] + '/diag_proc_block_mapped.txt'
    mapping(file_proc, file_code, file_out)
