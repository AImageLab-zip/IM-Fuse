import pandas as pd
from pathlib import Path
import shutil

input_paths = [Path(__file__).parent/path for path in ['test15splits.csv','val15splits.csv']]
output_paths = [Path(__file__).parent/path for path in ['test.csv','val.csv']]
input_train = Path(__file__).parent/'train.txt'
output_train = Path(__file__).parent/'train.csv'
for (input_path,output_path) in zip(input_paths,output_paths):
    df = pd.read_csv(input_path)
    subjs = df['case'].to_list()
    subjs = [sub + '\n' for sub in subjs]
    with open(output_path,'w') as f:
        f.writelines(subjs)
        
shutil.copy(input_train,output_train)