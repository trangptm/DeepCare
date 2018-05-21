__author__ = 'phtra'

from collections import  OrderedDict
import cPickle
import build_dict
import sys

def is_header(line):
    if 'a' <= line[0] <= 'z': return True
    return False

def calc_time(time_str, disease):
    # format of time - diabetes: 1/5/2002 02:16:42
    # format of time - mental  : 2002-05-28 16:49:00.000
    times = time_str.split()
    if len(times) < 2: return -1

    months = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    leap_year = [2004, 2008, 2012]

    year, month, day = 0, 0, 0
    if disease == 'diabetes':
        day, month, year = times[0].split('/')
    elif disease == 'mental':
        year, month, day = times[0].split('-')

    year, month, day = int(year), int(month), int(day)
    hours = int(times[1].split(':')[0])

    time = (year - 2001) * 365
    for i in range(month):
        time += months[i]
    time += day - 1
    time = time * 24 + hours

    for y in leap_year:
        if year > y: time += 24
        elif year == y:
            if month > 4: time += 24

    return time

def create_adm_dataset(diag_proc_file, medi_file, adm_file,
                       diag_dict, proc_dict, medi_dict,
                       adm_pkl, disease):
    n_diag = len(diag_dict)
    n_proc = len(proc_dict)

    prvsp_dict, prcae_dict = build_dict.build_adm_code_dict(adm_file)
    n_adm = len(prvsp_dict)

    adm_dataset = OrderedDict()
    list_col = ['patnt', 'admit_time', 'disch_time', 'method']
    for col in list_col:
        adm_dataset[col] = []

    adm_dataset['diag'] = [[] for i in range(n_adm)]
    adm_dataset['pm'] = [[] for i in range(n_adm)]
    adm_dataset['words'] = [[] for i in range(n_adm)]

    f_adm = open(adm_file, 'r')
    for line in f_adm:
        if is_header(line): continue
        ls = line.split('\t')

        adm_dataset['patnt'].append(ls[0])
        adm_dataset['admit_time'].append(calc_time(ls[3], disease))
        adm_dataset['disch_time'].append(calc_time(ls[4], disease))
        adm_dataset['method'].append('r' if ls[5][0] == 'X' else 'e')

    f_dpr = open(diag_proc_file, 'r')
    for line in f_dpr:
        if is_header(line): continue
        ls = line.split('\t')
        if ls[1] not in prcae_dict: continue

        idx = prcae_dict[ls[1]]

        if ls[3] == 'diagn':
            adm_dataset['diag'][idx].append(diag_dict[ls[6]])
            adm_dataset['words'][idx].append(diag_dict[ls[6]])
        else:
            adm_dataset['pm'][idx].append(proc_dict[ls[6]] + n_diag)
            adm_dataset['words'][idx].append(proc_dict[ls[6]] + n_diag)

    f_medi = open(medi_file, 'r')
    for line in f_medi:
        if is_header(line): continue
        ls = line.split('\t')
        if ls[1] not in prvsp_dict: continue

        idx = prvsp_dict[ls[1]]
        adm_dataset['pm'][idx].append(medi_dict[ls[5]] + n_diag + n_proc)
        adm_dataset['words'][idx].append(medi_dict[ls[5]] + n_diag + n_proc)

    f_pkl = open(adm_pkl, 'wb')
    cPickle.dump(adm_dataset, f_pkl, -1)

def create_atd_dataset(atd_file, diag_dict, atd_pkl, disease):
    f_atd = open(atd_file, 'r')
    list_col = ['ur', 'arr_time', 'dep_time', 'code']
    atd_dataset = OrderedDict()
    for col in list_col:
        atd_dataset[col] = []

    for line in f_atd:
        if is_header(line): continue

        ls = line.split('\t')
        atd_dataset['ur'].append(ls[1])
        atd_dataset['arr_time'].append(calc_time(ls[2], disease))
        atd_dataset['dep_time'].append(calc_time(ls[3], disease))
        atd_dataset['code'].append(diag_dict[ls[5]])

    f_pkl = open(atd_pkl, 'wb')
    cPickle.dump(atd_dataset, f_pkl, -1)

def main(disease = ''):

    diag_proc_file = 'preprocessed/' + disease + '/diag_proc_filtered.txt'
    medi_file = 'preprocessed/' + disease + '/medications_mapped_cutoff.txt'
    adm_file = 'preprocessed/' + disease + '/admissions_filtered.txt'
    atd_file = 'preprocessed/' + disease + '/atd_filtered.txt'

    adm_pkl = '../deepcare_data/' + disease + '/adm.pkl'
    atd_pkl = '../deepcare_data/' + disease + '/atd.pkl'
    dict_file = '../deepcare_data/' + disease + '/dict.pkl'


    diag_dict = build_dict.build_diag_dict(diag_proc_file, atd_file)
    proc_dict = build_dict.build_proc_dict(diag_proc_file)
    medi_dict = build_dict.build_medi_dict(medi_file)

    f = open(dict_file, 'wb')
    cPickle.dump((diag_dict, proc_dict, medi_dict), f, -1)

    create_adm_dataset(diag_proc_file, medi_file, adm_file,
                       diag_dict, proc_dict, medi_dict,
                       adm_pkl, disease)

    create_atd_dataset(atd_file, diag_dict, atd_pkl, disease)

if __name__ == '__main__':
    main(sys.argv[1])