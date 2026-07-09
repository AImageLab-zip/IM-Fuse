import argparse
from pathlib import Path

input_folder = Path('/work/phd_mimose')
for sub_dir in sorted(input_folder.iterdir()):
    if sub_dir.is_dir():
        n_elements = len(list(sub_dir.iterdir()))
        print(f'{sub_dir.name}: {n_elements}')
