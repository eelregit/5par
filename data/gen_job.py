#!/usr/bin/env python

import os
import sys


i_start, i_stop = int(sys.argv[1]), int(sys.argv[2])

job_script = f'job_{i_start}_{i_stop}.sh'
with open('job_template.sh', 'r') as f, open(job_script, 'w') as fo:
    job = f.read()
    job = job.format(i_start=i_start, i_stop=i_stop)
    fo.write(job)

os.system(f'sbatch {job_script}')
