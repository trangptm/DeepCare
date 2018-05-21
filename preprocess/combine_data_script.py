__author__ = 'phtra'

import sys
import os

disease = sys.argv[1]

list_steps = ['combine_data.py',
                 'create_patnt_records.py']

for step in list_steps:
    command = 'python ' + step + ' ' + disease
    print command
    os.system(command)