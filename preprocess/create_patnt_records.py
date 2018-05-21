__author__ = 'phtra'

from collections import OrderedDict
import cPickle
import numpy
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

def build_patnt_death(inp_file, disease):
    fi = open(inp_file, 'r')
    death = []

    for line in fi:
        if is_header(line): continue
        ls = line.split('\t')
        death.append(calc_time(ls[4], disease))

    return death

def build_patnt_dict(inp_file):
    patnt_dict = OrderedDict()
    ur_dict = OrderedDict()

    fi = open(inp_file, 'r')
    n_patnt = 0
    for line in fi:
        if is_header(line): continue

        ls = line.split('\t')
        patnt_dict[ls[0]] = n_patnt
        ur_dict[ls[1]] = n_patnt

        n_patnt += 1

    return patnt_dict, ur_dict

def map_adm(patnt_dict, adm_pkl):
    f = open(adm_pkl, 'rb')
    adm_dataset = cPickle.load(f)
    time_order = numpy.argsort(adm_dataset['admit_time'])

    n_patnt = len(patnt_dict)
    list_adm = [[] for i in range(n_patnt)]

    for i in time_order:
        if len(adm_dataset['diag'][i]) == 0: continue
        patnt_idx = patnt_dict[adm_dataset['patnt'][i]]
        list_adm[patnt_idx].append(i)

    return list_adm

def map_atd(ur_dict, atd_pkl):
    f = open(atd_pkl, 'rb')
    atd_dataset = cPickle.load(f)
    time_order = numpy.argsort(atd_dataset['arr_time'])

    n_patnt = len(ur_dict)
    list_atd = [[] for i in range(n_patnt)]

    for i in time_order:
        ur_idx = ur_dict[atd_dataset['ur'][i]]
        list_atd[ur_idx].append(i)

    return list_atd

def main(disease=''):

    patnt_file = 'N:/ResearchGroups/PRaDA/phtra/data/diabetes_mental/preprocessed/' + disease + '/patnts_filtered.txt'
    adm_pkl = '../deepcare_data/' + disease + '/adm.pkl'
    atd_pkl = '../deepcare_data/' + disease + '/atd.pkl'

    patnt_pkl = '../deepcare_data/' + disease + '/patnt.pkl'

    patnt_dataset = OrderedDict()
    patnt_dict, ur_dict = build_patnt_dict(patnt_file)
    death_date = build_patnt_death(patnt_file, disease)

    patnt_dataset['list_adm'] = map_adm(patnt_dict, adm_pkl)
    patnt_dataset['list_atd'] = map_atd(ur_dict, atd_pkl)
    patnt_dataset['death'] = death_date

    ff = open('death.txt', 'w')
    for date in death_date:
        ff.write('%d\n' % date)
    ff.close()

    f = open(patnt_pkl, 'wb')
    cPickle.dump(patnt_dataset, f, -1)


if __name__ == '__main__':
    main(disease=sys.argv[1])