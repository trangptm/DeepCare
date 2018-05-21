__author__ = 'phtra'

import sys

def cut_off_diag(inp_file, out_file, size):
    fi = open(inp_file, 'r')
    fo = open(out_file, 'w')
    for line in fi:
        ls = line.split('\t')
        if ls[0] == 'dgpro_refno':
            fo.write(line)
            continue

        if ls[2] in ['pccl', 'mdc']:
            continue

        if ls[3] == 'diagn':
            ls[6] = ls[6][:size]

        fo.write('\t'.join(ls))

def cut_off_medi(inp_file, out_file, size):
    fi = open(inp_file, 'r')
    fo = open(out_file, 'w')
    for line in fi:
        ls = line.split('\t')
        if ls[0] == 'prescriptionid':
            fo.write(line)
            continue

        codes = ls[5].split(',')
        for i in range(len(codes)):
            codes[i] = codes[i][:size]

        ls[5] = ','.join(codes)
        if len(ls[5]) <= 1: continue

        fo.write('\t'.join(ls))

def main(disease = '', diag_size = 2, medi_size = 6):

    diag_in = 'preprocessed/' + disease + '/diag_proc_block_mapped.txt'
    diag_out = 'preprocessed/' + disease + '/diag_proc_block_mapped_cutoff.txt'

    medi_in = 'preprocessed/' + disease + '/medications_mapped.txt'
    medi_out = 'preprocessed/' + disease + '/medications_mapped_cutoff.txt'

    cut_off_diag(diag_in, diag_out, diag_size)
    cut_off_medi(medi_in, medi_out, medi_size)

if __name__ == '__main__':
    main(disease=sys.argv[1])