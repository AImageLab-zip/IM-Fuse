"""Build 5 non-overlapping (cross-validation) splits for brats18/brats23/brats25 with a
CONSTANT train size across every fold and full test coverage.

Design goals (in priority order):
  1. train has the SAME number of samples in every fold: train = floor(TRAIN_FRAC * total)
     (round down -- "approssima per difetto"). e.g. brats18 -> 199, brats23 -> 875.
  2. Patient integrity: all timepoints of a patient stay in the same subset of a fold.
  3. The 5 folds' TEST sets are disjoint AND together cover 100% of the cases. Because a
     dataset's sample total is not divisible by 5, the `total % 5` remainder is spread
     across the test sets, so a few folds have one extra test sample (test differs by <=1).
  4. brats23 fold 1 reproduces old_split.json['brats23'] verbatim (its train/val/test and
     val masks). That fold naturally carries brats23's +1 test remainder (old test = 251).

Sizes per fold:
    train = floor(TRAIN_FRAC * total)      (identical across folds)
    test  = total // 5  (+1 for the first `total % 5` folds; brats23 fold 1 pinned to old)
    val   = total - train - test           (the remainder; differs by at most 1)

Sizes are computed at the *sample* (timepoint / sub-id) level, not the patient level, so
multi-timepoint patients do not skew the counts.

Dropped constraint (relative to the original version): brats25 fold k no longer has to
CONTAIN brats23 fold k. Each dataset is split independently.

Stratification is best-effort: test bins are filled from a class-stratified ordering of
single-timepoint patients (round-robin), so each bin gets a spread of classes, but the
target sizes are always honoured first.

Stratification keys:
  * brats18/brats23 -> `Site No` + `Cohort Name` from BraTS2023_2017_GLI_Mapping.xlsx
  * brats25         -> the PTG demographic composite key (falls back to 'unknown').

Output: data/splits/splits_5fold.json with the fold index (1..5) at the top level.
This does NOT overwrite the live split.json.
"""
import json
import os
import re
import copy
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
from itertools import product

# --------------------------------------------------------------------------- config
SPLITS_DIR = Path(__file__).parent.parent / 'mimose' / 'data' / 'splits'
# the brats25 case directories live here; override with MIMOSE_UNPACKED_DIR for testing
UNPACKED_DIR = os.environ.get('MIMOSE_UNPACKED_DIR', '/work/phd_mimose/unpacked')
RANDOM_STATE = 42
N_FOLDS = 5
SUPPORTED_DATASETS = ['brats18', 'brats23', 'brats25']
THREE_SPLITS = ['train', 'val', 'test']
TRAIN_FRAC = 0.70  # target train fraction of the whole dataset (floored)

SITE_COL = 'Site No (represents the originating institution)'
COHORT_COL = 'Cohort Name (if publicly available)'

# --------------------------------------------------------------------------- helpers
PREFIX_RE = re.compile(r'^(BraTS-GLI-\d+)-')

# all 15 non-empty subsets of the 4 modalities (flair, t1ce, t1, t2)
MODALITY_COMBOS = [list(c) for c in product([False, True], repeat=4) if any(c)]


def patient_prefix(sub) -> str:
    sub = str(sub)
    m = PREFIX_RE.match(sub)
    return m.group(1) if m else sub


def stratified_order(patients, key_of, rng):
    """Deterministic ordering of `patients` that interleaves stratification classes:
    group by key, shuffle within each group, then round-robin across groups. Dealing
    this order into bins gives every bin a similar class distribution."""
    groups = defaultdict(list)
    for p in patients:
        groups[key_of(p)].append(p)
    glists = sorted(groups.values(), key=lambda g: (-len(g), key_of(g[0])))
    for g in glists:
        rng.shuffle(g)
    ordered = []
    max_len = max((len(g) for g in glists), default=0)
    for i in range(max_len):
        for g in glists:
            if i < len(g):
                ordered.append(g[i])
    return ordered


def build_entries(patient_subset, patient_subs, rng):
    """patient_subset: {patient -> 'train'|'val'|'test'}. Returns {split: [ {sub, mask}, ... ]}.
    val entries receive a random non-empty modality mask; train/test masks are None."""
    out = {s: [] for s in THREE_SPLITS}
    for patient, subset in patient_subset.items():
        for sub in patient_subs.get(patient, []):
            entry = {'sub': sub, 'mask': None}
            if subset == 'val':
                entry['mask'] = list(rng.choice(MODALITY_COMBOS))
            out[subset].append(entry)
    return out


def build_test_bins(patients, subs_of, n_folds, key_of, rng, fixed0=None):
    """Partition ALL `patients` into `n_folds` disjoint test bins covering everything.

    Bin sample loads are balanced to floor/ceil of the per-bin target. If `fixed0` is
    given (a set of patients), bin 0 is pinned to exactly those patients and the rest are
    balanced across the remaining bins. Returns (bins, load)."""
    bins = [[] for _ in range(n_folds)]
    load = [0] * n_folds

    remaining = list(patients)
    active = list(range(n_folds))
    if fixed0 is not None:
        bins[0] = list(fixed0)
        load[0] = sum(subs_of[p] for p in fixed0)
        remaining = [p for p in patients if p not in fixed0]
        active = list(range(1, n_folds))

    rem_total = sum(subs_of[p] for p in remaining)
    base, extra = divmod(rem_total, len(active))
    targets = [0] * n_folds
    for j, i in enumerate(active):
        targets[i] = base + (1 if j < extra else 0)

    multi = sorted((p for p in remaining if subs_of[p] > 1), key=lambda p: (-subs_of[p], p))
    singles = stratified_order([p for p in remaining if subs_of[p] == 1], key_of, rng)

    # multi-timepoint patients: into the bin with the most remaining room (keep coverage)
    for p in multi:
        s = subs_of[p]
        room = [i for i in active if load[i] + s <= targets[i]]
        i = (min(room, key=lambda i: (load[i], i)) if room
             else min(active, key=lambda i: (load[i] - targets[i], i)))
        bins[i].append(p)
        load[i] += s

    # single-timepoint patients: round-robin fill up to each target (stratified order)
    si = 0
    while si < len(singles) and any(load[i] < targets[i] for i in active):
        progressed = False
        for i in active:
            if si >= len(singles):
                break
            if load[i] < targets[i]:
                bins[i].append(singles[si]); load[i] += 1; si += 1; progressed = True
        if not progressed:
            break
    # any leftover singles (only if a multi overshot a target) -> least-loaded bin
    while si < len(singles):
        i = min(active, key=lambda i: load[i])
        bins[i].append(singles[si]); load[i] += 1; si += 1

    placed = sum(len(b) for b in bins)
    if placed != len(patients) or sum(load) != sum(subs_of.values()):
        raise RuntimeError('test bin construction did not cover all patients')
    return bins, load


def make_folds(patient_subs, key_of, n_folds, train_frac, seed, verbatim_fold0=None):
    """Build `n_folds` folds with a constant train size and full, disjoint test coverage.

    If `verbatim_fold0` (a {split: [entries]} dict) is given, fold 1 is reproduced from it
    verbatim and its test patients are pinned as fold-1's test bin.
    Returns (folds, sizes)."""
    patients = sorted(patient_subs)
    subs_of = {p: len(patient_subs[p]) for p in patients}
    total = sum(subs_of.values())
    train_n = int(train_frac * total)  # floor

    rng = random.Random(seed)
    fixed0 = None
    if verbatim_fold0 is not None:
        fixed0 = {patient_prefix(e['sub']) for e in verbatim_fold0['test']}
    bins, load = build_test_bins(patients, subs_of, n_folds, key_of, rng, fixed0)

    mask_rng = random.Random(seed)
    folds = []
    for k in range(n_folds):
        if k == 0 and verbatim_fold0 is not None:
            folds.append(copy.deepcopy(verbatim_fold0))
            continue
        test_patients = set(bins[k])
        remainder = [p for p in patients if p not in test_patients]
        val_n = (total - load[k]) - train_n  # what's left after test and the fixed train
        rem_singles = [p for p in remainder if subs_of[p] == 1]
        random.Random(seed + 1000 + k).shuffle(rem_singles)
        if val_n < 0 or len(rem_singles) < val_n:
            raise RuntimeError(
                f'fold {k + 1}: cannot form a val set of {val_n} from '
                f'{len(rem_singles)} single-timepoint patients')
        val_patients = set(rem_singles[:val_n])

        subset_of = {p: ('val' if p in val_patients else 'train') for p in remainder}
        subset_of.update({p: 'test' for p in test_patients})
        folds.append(build_entries(subset_of, patient_subs, mask_rng))

    return folds, {'train': train_n, 'test_min': min(load), 'test_max': max(load),
                   'total': total}


# --------------------------------------------------------------------------- load inputs
with open(SPLITS_DIR / 'old_split.json') as f:
    old_split = json.load(f)

mapping = pd.read_excel(SPLITS_DIR / 'BraTS2023_2017_GLI_Mapping.xlsx')
unpacked_dir_list = [p.name for p in Path(UNPACKED_DIR).iterdir()]

# ---- strat key for brats18/brats23: Site No + Cohort Name, keyed by patient prefix ----
mapping['_prefix'] = mapping['BraTS2023'].astype(str).map(patient_prefix)
b23_key = {}
for _, row in mapping.iterrows():
    b23_key.setdefault(row['_prefix'], f"{row[SITE_COL]}|{row[COHORT_COL]}")


def b23_key_of(prefix):
    return b23_key.get(prefix, 'unknown')


# ---- strat key for brats25 cases: PTG demographic composite, keyed by prefix ----
ptg_meta = pd.read_excel(
    SPLITS_DIR / 'BraTS-PTG supplementary demographic information and metadata.xlsx'
)
ptg_meta['Glioma type '] = ptg_meta['Glioma type '].str.strip().str.title()
ptg_meta['age_bin'] = pd.cut(
    ptg_meta["Patient's Age"], bins=[0, 40, 60, 200], labels=['<40', '40-60', '>60']
)
ptg_meta['_prefix'] = ptg_meta['BraTS Subject ID'].astype(str).map(patient_prefix)
ptg_key = {}
for prefix, rows in ptg_meta.groupby('_prefix'):
    r = rows.iloc[0]
    ptg_key[prefix] = '|'.join(str(r[c]) for c in [
        'Site', 'Magnetic Field Strength', 'Manufacturer',
        "Patient's Sex", 'Glioma type ', 'age_bin',
    ])


def ptg_key_of(prefix):
    return ptg_key.get(prefix, 'unknown')


# --------------------------------------------------------------------------- patient -> subs
# brats23: patient -> list of its brats23 sub ids (from old_split, all subsets pooled)
b23_patient_subs = {}
for split in THREE_SPLITS:
    for e in old_split['brats23'][split]:
        b23_patient_subs.setdefault(patient_prefix(e['sub']), []).append(e['sub'])

# brats18: remap old brats18 subs -> BraTS2023 ids, keyed by prefix
b18_patient_subs = {}
for split in THREE_SPLITS:
    for e in old_split['brats18'][split]:
        row = mapping.loc[mapping['BraTS2018'] == e['sub'], 'BraTS2023']
        if len(row) == 0:
            continue
        new_sub = row.iloc[0]
        b18_patient_subs.setdefault(patient_prefix(new_sub), []).append(new_sub)

# brats25: every unpacked case dir grouped by patient prefix
b25_patient_subs = {}
for d in unpacked_dir_list:
    b25_patient_subs.setdefault(patient_prefix(d), []).append(d)

# --------------------------------------------------------------------------- build folds
brats18_folds, b18_sizes = make_folds(
    b18_patient_subs, b23_key_of, N_FOLDS, TRAIN_FRAC, RANDOM_STATE)
brats23_folds, b23_sizes = make_folds(
    b23_patient_subs, b23_key_of, N_FOLDS, TRAIN_FRAC, RANDOM_STATE,
    verbatim_fold0=old_split['brats23'])
brats25_folds, b25_sizes = make_folds(
    b25_patient_subs, ptg_key_of, N_FOLDS, TRAIN_FRAC, RANDOM_STATE)

folds_by_dataset = {'brats18': brats18_folds, 'brats23': brats23_folds, 'brats25': brats25_folds}
sizes_by_dataset = {'brats18': b18_sizes, 'brats23': b23_sizes, 'brats25': b25_sizes}
combined = {
    str(k + 1): {ds: folds_by_dataset[ds][k] for ds in SUPPORTED_DATASETS}
    for k in range(N_FOLDS)
}

# --------------------------------------------------------------------------- checks & report
violations = []


def subset_prefixes(entries):
    return {patient_prefix(e['sub']) for e in entries}


for k in range(N_FOLDS):
    fold = combined[str(k + 1)]
    print(f'\n=== fold {k + 1} ===')
    for ds in SUPPORTED_DATASETS:
        counts = {s: len(fold[ds][s]) for s in THREE_SPLITS}
        total = sum(counts.values()) or 1
        print(f'  {ds}: ' + ', '.join(
            f'{s} {counts[s]} ({counts[s] / total * 100:.1f}%)' for s in THREE_SPLITS))

        # patient integrity: no patient prefix spans two subsets within this fold
        prefix_to_splits = {}
        for s in THREE_SPLITS:
            for prefix in subset_prefixes(fold[ds][s]):
                prefix_to_splits.setdefault(prefix, set()).add(s)
        bad = {p: s for p, s in prefix_to_splits.items() if len(s) > 1}
        if bad:
            violations.append(f'fold {k + 1} {ds}: patients span splits {bad}')

# constant train size across folds
for ds in SUPPORTED_DATASETS:
    trains = [len(combined[str(k + 1)][ds]['train']) for k in range(N_FOLDS)]
    if len(set(trains)) != 1 or trains[0] != sizes_by_dataset[ds]['train']:
        violations.append(f'{ds}: train size not constant across folds: {trains}')

# 5-fold property: per dataset, test sets disjoint and their union covers all cases
for ds in SUPPORTED_DATASETS:
    test_sets = [{e['sub'] for e in combined[str(k + 1)][ds]['test']} for k in range(N_FOLDS)]
    union = set().union(*test_sets)
    total_overlap = sum(len(test_sets[i] & test_sets[j])
                        for i in range(N_FOLDS) for j in range(i + 1, N_FOLDS))
    all_cases = {e['sub'] for k in range(N_FOLDS) for s in THREE_SPLITS
                 for e in combined[str(k + 1)][ds][s]}
    if total_overlap:
        violations.append(f'{ds}: test sets overlap ({total_overlap} shared cases)')
    if union != all_cases:
        violations.append(f'{ds}: test folds do not cover all cases '
                          f'({len(all_cases - union)} never tested)')
    # test sizes differ by at most 1 (spread remainder)
    sizes = sorted(len(t) for t in test_sets)
    if sizes[-1] - sizes[0] > 1:
        violations.append(f'{ds}: test sizes differ by more than 1: {sizes}')

# fold-1 brats23 must equal the preexisting split exactly
if combined['1']['brats23'] != old_split['brats23']:
    violations.append('fold 1 brats23 does not match old_split.json exactly')

print('\n--- per-fold sizes ---')
for ds in SUPPORTED_DATASETS:
    s = sizes_by_dataset[ds]
    test_desc = (str(s['test_min']) if s['test_min'] == s['test_max']
                 else f"{s['test_min']}-{s['test_max']}")
    print(f'  {ds}: train {s["train"]} (constant), test {test_desc}, '
          f'val = rest (total {s["total"]})')

if violations:
    print('\n[FAIL]')
    for v in violations:
        print('  - ' + v)
    raise RuntimeError(f'{len(violations)} split constraint(s) violated')
print('\n[OK] all constraints satisfied '
      '(constant train, fold-1 brats23 exact, disjoint+covering test folds, patient integrity)')

output_path = SPLITS_DIR / 'splits_5fold.json'
with open(output_path, 'w') as f:
    json.dump(combined, f, indent=2)
print(f'Saved to {output_path}')
