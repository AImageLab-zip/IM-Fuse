import pandas as pd
input_path = '/homes/ocarpentiero/IM-Fuse/MaM/datalist/test15splits.csv'
output_path = '/homes/ocarpentiero/IM-Fuse/MaM/datalist/test.txt'

df = pd.read_csv(input_path)
sub_list = df['case'].to_list()
with open(output_path,'w') as f:
    f.writelines([sub + '\n' for sub in sub_list])