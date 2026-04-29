import json
from pathlib import Path
import pandas as pd
import ast

input_dir = Path('/home/ocarpentiero/PycharmProjects/IM-Fuse/src/mimose/data/splits')
output_18 = input_dir / 'split.json'
#(flair, t1ce, t1, t2)
modes = ['brats18','brats23']
final_dict= { key:{

    'train': [],
    'val': [],
    'test':[]
} for key in modes
}

for mode in modes:
    with open(input_dir / mode / 'train.txt') as f:
        for line in f:
            line = line.strip()
            final_dict[mode]['train'].append(
                {
                    'sub':line,
                    'mask':None
                }
            )

    df = pd.read_csv(input_dir/mode/'val.csv')
    for row in df.iterrows():
        sub, str_mask = row[1].to_list()
        mask = ast.literal_eval(str_mask)
        sorted_mask = [mask[1],mask[2],mask[0],mask[3]]

        final_dict[mode]['val'].append(
            {
                'sub': sub,
                'mask': sorted_mask
            }
        )

    with open(input_dir / mode/ 'test.txt') as f:
        for line in f:
            line = line.strip()
            final_dict[mode]['test'].append(
                {
                    'sub':line,
                    'mask':None
                }
            )


with open(output_18,'w') as f:
    json.dump(final_dict,f,indent=3)
