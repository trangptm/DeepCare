__author__ = 'phtra'

import sys
import os

disease = sys.argv[1]

list_steps = ['raw_preprocess.py',
                 'map_medi_code.py',
                 'map_proc_code.py',
                 'cut_off_code.py',
                 'filter_adm.py',
                 'filter_cutoff_atd.py',
                 'filter_patients.py']

for step in list_steps:
    command = 'python ' + step + ' ' + disease
    print command
    os.system(command)