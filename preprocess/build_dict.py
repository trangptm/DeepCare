__author__ = 'ptmin'

from collections import OrderedDict

def is_header(line):
    if 'a' <= line[0] <= 'z': return True
    return False

def build_diag_dict(diag_file, atd_file):
    diag_dict = OrderedDict()
    n_words = 0

    f_diag = open(diag_file, 'r')
    f_atd = open(atd_file, 'r')

    for line in f_diag:
        ls = line.split('\t')

        #ls[3]: diagn or proce, ls[6]: code
        if ls[3] == 'diagn':
            if ls[6] not in diag_dict:
                diag_dict[ls[6]] = n_words
                n_words += 1

    for line in f_atd:
        if is_header(line): continue
        ls = line.split('\t')

        if ls[5] not in diag_dict:
            diag_dict[ls[5]] = n_words
            n_words += 1

    return diag_dict

def build_proc_dict(proc_file):
    proc_dict = OrderedDict()
    n_words = 0

    f_proc = open(proc_file, 'r')

    for line in f_proc:
        ls = line.split('\t')
        if ls[3] == 'proce':
            if ls[6] not in proc_dict:
                proc_dict[ls[6]] = n_words
                n_words += 1

    return proc_dict

def build_medi_dict(medi_file):
    medi_dict = OrderedDict()
    n_words = 0

    f_medi = open(medi_file, 'r')
    for line in f_medi:
        if is_header(line): continue
        ls = line.split('\t')

        if ls[5] not in medi_dict:
            medi_dict[ls[5]] = n_words
            n_words += 1

    return medi_dict

def build_adm_code_dict(adm_file):
    # build 2 dicts: prvsp_dict & prcae_dict

    prvsp_dict = OrderedDict()
    prcae_dict = OrderedDict()

    f_adm = open(adm_file, 'r')
    n_adm = 0
    for line in f_adm:
        if is_header(line): continue
        ls = line.split('\t')

        prvsp_dict[ls[1]] = n_adm
        prcae_dict[ls[2]] = n_adm

        n_adm += 1

    return prvsp_dict, prcae_dict

if __name__ == '__main__':
    print is_header('abc\n')
    print is_header('123\t')