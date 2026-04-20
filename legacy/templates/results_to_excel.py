import pandas as pd
from pathlib import Path

results_path = Path('/homes/ocarpentiero/results/test_lckd_18.txt')
excel_path = results_path.with_suffix('.xlsx')
def parse_line(s: str) -> dict:
    left, right = s.split("-->")
    modals = left.split("=")[1].strip()

    metrics = {}
    for item in right.split(","):
        key, value = item.split("=")
        metrics[key.strip()] = float(value.strip())

    return {"modals": modals, **metrics}
def parse_avg(s: str) -> dict:
    left, right = s.split("-->")

    metrics = {}
    for item in right.split(","):
        key, value = item.split("=")
        metrics[key.strip()] = float(value.strip())

    return metrics

def string_to_code(string):
    code = 0
    if 't1c' in string:
        code+=1
    if 't1n' in string:
        code+=2
    if 't2f' in string:
        code+=4
    if 't2w' in string:
        code+=8
    return code

def string_to_order(string):

    code = string_to_code(string)
    match code:
        case 1:
            return 2
        case 2:
            return 1
        case 3:
            return 7
        case 4:
            return 0
        case 5:
            return 5
        case 6:
            return 4
        case 7:
            return 10
        case 8:
            return 3
        case 9:
            return 9
        case 10:
            return 8
        case 11:
            return 13
        case 12:
            return 6
        case 13:
            return 12
        case 14:
            return 11
        case 15:
            return 14
        case _:
            raise RuntimeError(f'invalid code passed:{string}, {code}')

with open(results_path, 'r') as f:
    scores = {
        'ET':[],
        'TC':[],
        'WT':[],
        'order':[]
    }
    for line in f:
        if 'Avg' in line:
            parsed = parse_avg(line)
            scores['order'].append(15)
            scores['ET'].append(parsed['ET'])
            scores['WT'].append(parsed['WT'])
            scores['TC'].append(parsed['TC'])
        else:
            parsed = parse_line(line)
            scores['order'].append(string_to_order(parsed['modals']))
            scores['ET'].append(parsed['ET']*100)
            scores['WT'].append(parsed['WT']*100)
            scores['TC'].append(parsed['TC']*100)

    scores_sorted = {
        'ET':[],
        'TC':[],
        'WT':[]
    }
for key,_ in scores_sorted.items():
    scores_sorted[key] = [v for _, v in sorted(zip(scores['order'], scores[key]))]

for key,s in scores_sorted.items():
    s[-1] = sum(s[:-1])/len(s[:-1])
df = pd.DataFrame(scores_sorted)
df.to_excel(excel_path,index=False)