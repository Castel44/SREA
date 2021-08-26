import os
import subprocess

######################################################################################################################

BASE_PATH = subprocess.check_output('git rev-parse --show-toplevel', shell=True).decode('ascii').split('\n', 1)[0]
OUTPATH = os.path.join(BASE_PATH, 'results')
