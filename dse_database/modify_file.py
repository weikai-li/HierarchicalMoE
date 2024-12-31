from glob import iglob
from pprint import pprint
from os import rename

prev_name = 'fdtd_'
new_name = 'fdtd-2d_'
path = './poly/databases/**/*'

files = [f for f in iglob(path, recursive=True) if f.endswith('.db') and prev_name in f]
pprint(files)
for f in files:
    rename(f, f.replace(prev_name, new_name))