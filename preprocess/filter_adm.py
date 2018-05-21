__author__ = 'phtra'

# build a list of all admissions those are not dialysis
# filter out all diag_prog, admissions related to dialysis

import re
import sys
from collections import OrderedDict

def build_adm_list(inp_file):
    # build a list of all admissions: value 0: routine adm, value 1: emergency adm, value -1: noise adm
    fi = open(inp_file, 'r')
    list_admi = OrderedDict()

    for line in fi:
        ls = line.lower().split('\t')
        #ls[2]: prcae_refno -> keys
        #ls[5]: types of admissions
        #value: 0: routine, 1: emergency, -1: noise
        key = ls[2]
        value = -1
        if ls[5][0] == 'x': value = 0
        elif ls[5][0] in 'co': value = 1
        list_admi[key] = value

    return list_admi

def filter_dialysis(inp_file, list_admi):
    # from diag_proc file, find all routine adm with dialysis
    fi = open(inp_file, 'r')

    for line in fi:
        ls = line.lower().split('\t')
        key = ls[1]
        if re.search(r'dialysis', ls[7]) and (key in list_admi):
            if list_admi[key] == 0:
                list_admi[key] = -1

    return list_admi

def filter_diag_proc(inp_file, out_file, list_admi):
    fi = open(inp_file, 'r')
    fo = open(out_file, 'w')

    for line in fi:
        ls = line.split('\t')
        if (ls[0] == 'dgpro_refno'):
            fo.write(line)
            continue

        if (ls[1] in list_admi and list_admi[ls[1]] != -1):
            if (ls[3] == 'diagn') and (ls[6][0] <= '9' and ls[6][0] >= '0'): continue
            if ls[6] != '0':
                fo.write(line)

    fo.close()

def filter_admi(inp_file, out_file, list_admi):
    fi = open(inp_file, 'r')
    fo = open(out_file, 'w')

    for line in fi:
        ls = line.split('\t')
        if (ls[0] == 'patnt_refno') or (ls[2] in list_admi and list_admi[ls[2]] != -1):
            fo.write(line)

    fo.close()


def main(disease=''):

    diag_proc_file = 'preprocessed/' + disease + '/diag_proc_block_mapped_cutoff.txt'
    admi_file = 'preprocessed/' + disease + '/admissions.txt'
    out_admi_file = 'preprocessed/' + disease + '/admissions_filtered.txt'
    out_diag_proc_file = 'preprocessed/' + disease + '/diag_proc_filtered.txt'


    list_admi = build_adm_list(admi_file)
    list_admi = filter_dialysis(diag_proc_file, list_admi)

    filter_diag_proc(diag_proc_file, out_diag_proc_file, list_admi)
    filter_admi(admi_file, out_admi_file, list_admi)

if __name__ == '__main__':
    main(sys.argv[1])