import json
import re
from pathlib import Path
import pandas as pd
import copy
from itertools import cycle, product
from sklearn.model_selection import train_test_split

old_split_path = Path(Path(__file__).parent.parent)/'mimose'/'data'/'splits'/'old_split.json'
with open(old_split_path,'r') as f:
    old_split = json.load(f)
mapping = pd.read_excel(Path(__file__).parent.parent/'mimose'/'data'/'splits'/'BraTS2023_2017_GLI_Mapping.xlsx')

supported_datasets = ['brats18','brats23','brats25']
three_splits = ['train','val','test']
new_split = {
    dataset:{
        split:[] for split in three_splits
    }
    for dataset in supported_datasets
}


# BraTS18 part
for split in three_splits:
    for element in old_split['brats18'][split]:
        sub_name = element['sub']
        new_sub_name = mapping.loc[mapping["BraTS2018"] == sub_name, "BraTS2023"].iloc[0]
        new_element = copy.deepcopy(element)
        new_element['sub'] = new_sub_name
        new_split['brats18'][split].append(new_element)

# BraTS23 part
new_split['brats23'] = copy.deepcopy(old_split['brats23'])

unpacked_dir = '/work/grana_neuro/mimose/unpacked'
unpacked_dir_list = [path.name for path in Path(unpacked_dir).iterdir()]

# BraTS25 part
combinations = [
    list(combo)
    for combo in product([False, True], repeat=4)
    if any(combo)
]
rotating_modals = cycle(combinations)
brats25_seen = set()
for split in three_splits:
    for element in old_split['brats23'][split]:
        sub_name = element['sub']
        matches = [dir for dir in unpacked_dir_list if dir.startswith(sub_name[:-4]) and dir not in brats25_seen]
        for match in matches:
            brats25_seen.add(match)
            new_element = copy.deepcopy(element)
            new_element['sub'] = match
            if split == 'val':
                new_element['mask'] = next(rotating_modals)
            new_split['brats25'][split].append(new_element)

# BraTS25 part - remaining (unmatched) samples, stratified 70/10/20
matched_set = {match for split in three_splits for element in old_split['brats23'][split]
               for match in unpacked_dir_list if match.startswith(element['sub'][:-4])}
unmatched = [u for u in unpacked_dir_list if u not in matched_set]

patient_groups = {}
for u in unmatched:
    m = re.match(r'^(BraTS-GLI-\d+)-', u)
    prefix = m.group(1) if m else u
    patient_groups.setdefault(prefix, []).append(u)

ptg_meta = pd.read_excel(
    Path(Path(__file__).parent.parent) / 'mimose' / 'data' / 'splits' /
    'BraTS-PTG supplementary demographic information and metadata.xlsx'
)
ptg_meta['Glioma type '] = ptg_meta['Glioma type '].str.strip().str.title()
ptg_meta['age_bin'] = pd.cut(ptg_meta["Patient's Age"], bins=[0, 40, 60, 200],
                              labels=['<40', '40-60', '>60'])

group_prefixes = []
group_strat_keys = []
for prefix, samples in patient_groups.items():
    rows = ptg_meta[ptg_meta['BraTS Subject ID'].isin(samples)]
    if len(rows) == 0:
        key = 'unknown'
    else:
        r = rows.iloc[0]
        key = '|'.join(str(r[c]) for c in [
            'Site', 'Magnetic Field Strength', 'Manufacturer',
            "Patient's Sex", 'Glioma type ', 'age_bin'
        ])
    group_prefixes.append(prefix)
    group_strat_keys.append(key)

key_counts = pd.Series(group_strat_keys).value_counts()
max_unique_keys = int(0.10 * len(group_prefixes)) - 2  # must be < n_val in second split
top_keys = set(key_counts.nlargest(max_unique_keys - 1).index)  # -1 to leave room for 'rare'
group_strat_keys = [k if k in top_keys else 'rare' for k in group_strat_keys]

train_val_prefixes, test_prefixes, train_val_keys, _ = train_test_split(
    group_prefixes, group_strat_keys, test_size=0.20, random_state=42, stratify=group_strat_keys
)
train_prefixes, val_prefixes = train_test_split(
    train_val_prefixes, test_size=0.10 / 0.80, random_state=42, stratify=train_val_keys
)

total_unmatched_samples = sum(len(v) for v in patient_groups.values())
existing_val   = len(new_split['brats25']['val'])
existing_test  = len(new_split['brats25']['test'])
existing_train = len(new_split['brats25']['train'])
total_samples  = existing_train + existing_val + existing_test + total_unmatched_samples
target_val_samples  = round(0.10 * total_samples) - existing_val
target_test_samples = round(0.20 * total_samples) - existing_test

train_set = set(train_prefixes)
val_set   = set(val_prefixes)
test_set  = set(test_prefixes)

cur_val_samples  = sum(len(patient_groups[p]) for p in val_set)
cur_test_samples = sum(len(patient_groups[p]) for p in test_set)

for p in sorted(train_set, key=lambda p: len(patient_groups[p])):
    if cur_val_samples >= target_val_samples:
        break
    n = len(patient_groups[p])
    if abs(cur_val_samples + n - target_val_samples) < abs(cur_val_samples - target_val_samples):
        val_set.add(p)
        train_set.remove(p)
        cur_val_samples += n

for p in sorted(train_set, key=lambda p: len(patient_groups[p])):
    if cur_test_samples >= target_test_samples:
        break
    n = len(patient_groups[p])
    if abs(cur_test_samples + n - target_test_samples) < abs(cur_test_samples - target_test_samples):
        test_set.add(p)
        train_set.remove(p)
        cur_test_samples += n

split_assignment = {}
for p in train_set: split_assignment[p] = 'train'
for p in val_set:   split_assignment[p] = 'val'
for p in test_set:  split_assignment[p] = 'test'

for prefix, samples in patient_groups.items():
    assigned_split = split_assignment[prefix]
    for sample in samples:
        new_element = {'sub': sample, 'mask': None}
        if assigned_split == 'val':
            new_element['mask'] = next(rotating_modals)
        new_split['brats25'][assigned_split].append(new_element)


for dataset in supported_datasets:
    counts = {s: len(new_split[dataset][s]) for s in three_splits}
    total = sum(counts.values())
    print(f'{dataset} number of samples: ' + ', '.join(
        f'{s}: {counts[s]} ({counts[s]/total*100:.1f}%)' for s in three_splits
    ))

# Integrity check: all samples sharing the same patient prefix must be in the same split
violations = []
for dataset in supported_datasets:
    prefix_to_splits = {}
    for split in three_splits:
        for element in new_split[dataset][split]:
            m = re.match(r'^(BraTS-GLI-\d+)-', element['sub'])
            prefix = m.group(1) if m else element['sub']
            prefix_to_splits.setdefault(prefix, set()).add(split)
    bad = {p: s for p, s in prefix_to_splits.items() if len(s) > 1}
    if bad:
        violations.append((dataset, bad))
        for p, splits in bad.items():
            print(f'[INTEGRITY FAIL] {dataset}: patient {p} appears in {splits}')
    else:
        print(f'[INTEGRITY OK] {dataset}: all patients are confined to a single split')

if violations:
    raise RuntimeError(f'Patient integrity violated in {[d for d, _ in violations]}')

output_path = old_split_path.parent / 'split.json'
with open(output_path, 'w') as f:
    json.dump(new_split, f, indent=2)
print(f'Saved to {output_path}')