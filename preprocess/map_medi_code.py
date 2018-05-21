__author__ = 'phtra'

import re
from sets import Set
import sys

def mapping(file_code, file_medi, file_out):
    f_code = open(file_code, 'r')
    f_medi = open(file_medi, 'r')
    f_mapp = open(file_out, 'w')

    name_list = []
    code_list = []
    count = 0
    for line in f_code:
        ls = line.lower().split('\t')
        if ls[0] == 'code': continue

        name = re.sub(r'[(][^)]*[)]', '', ls[1])

        dose = re.sub(r'patches', 'patch', ls[2])
        dose = re.sub(r'(micrograms)', 'mcg', dose)
        dose = re.sub(r'(microgram)', 'mcg', dose)
        dose = re.sub(r's(\W|\Z)', ' ', dose)
        dose = re.sub(r'\W', ' ', dose)
        code_info = Set(name.split() + dose.split())

        # if count < 20: print code_info
        # count += 1

        name_list.append(code_info)
        code_list.append(ls[3])

    for line in f_medi:
        ls = line.lower().split('\t')
        if ls[0] == 'prescriptionid':
            f_mapp.write(line.lower())
            continue
        if len(ls) < 6: continue

        name = re.sub(r'mg\W', ' mg ' ,ls[5])
        name = re.sub(r'mcg\W', ' mcg ', name)
        name = re.sub(r'ml\W', ' ml ', name)
        name = re.sub(r'\W', ' ', name)

        name = re.sub(r'patches', 'patch', name)
        name = re.sub(r'pessaries', 'pessary', name)
        name = re.sub(r's\Z', '', name)

        name_set = Set(name.split())
        max_num = 0
        code = ''
        for i in range(len(name_list)):
            num_intersect = len(name_set.intersection(name_list[i]))
            if max_num < num_intersect:
                code = code_list[i]
                max_num = num_intersect

        ls[5] = code

        print count
        count += 1
        if len(code) > 0:
            f_mapp.write('\t'.join(ls))

    f_mapp.close()

if __name__ == '__main__':
    file_code = 'CODE/pbs_parsed_add_tab.txt'
    file_medi = 'preprocessed/' + sys.argv[1] + '/medications.txt'
    file_out = 'preprocessed/' + sys.argv[1] + '/medications_mapped.txt'
    mapping(file_code, file_medi, file_out)