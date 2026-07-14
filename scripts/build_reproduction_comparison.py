#!/usr/bin/env python3
"""Build a reproduction-comparison workbook from the results directory.

Reads the published `legacy-results.xlsx` and the per-fold result workbooks
produced for each model/dataset, then writes a single workbook with one tab per
*available* model. Each tab holds two tables (BRATS2018 and BRATS2023). Every
table lists the modality-presence columns followed, for each region (WT, TC,
ET) and each metric (Dice, HD95), by: the legacy value, one column per fold,
the mean and std of the folds, and an error column (legacy - mean of folds).

Note: the legacy workbook only contains Dice numbers, so the legacy (and hence
error) columns of the HD95 tables are left blank.
"""

from __future__ import annotations

import glob
import os
import re
import statistics
from collections import defaultdict

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from rich.progress import track

# --------------------------------------------------------------------------- #
# Hardcoded parameters
# --------------------------------------------------------------------------- #
RESULTS_DIR = "/work/phd_mimose/results"
LEGACY_FILE = "legacy-results.xlsx"
OUTPUT_FILE = "reproduction-comparison.xlsx"

# Regions to report, in output order. Keys are the fold-file column names for
# Dice; the HD95 column is "<name>_hd95". Legacy stores them under the labels in
# LEGACY_REGION_LABELS.
REGIONS = ["WT", "TC", "ET"]
LEGACY_REGION_LABELS = {
    "Whole Tumor": "WT",
    "Tumor Core": "TC",
    "Enhancing Tumor": "ET",
}

# Maps the numeric directory suffix to a dataset label.
DATASETS = {"18": "BRATS2018", "23": "BRATS2023"}

# Every model/dataset combination is expected to have these folds tested,
# per the sbatch_files/<model>/all layout.
EXPECTED_FOLDS = ["fold1", "fold3", "fold5"]

# Config templates are the source of truth for which models exist for a
# given dataset (one `<model>_<dsnum>.yaml` per model/dataset combination).
CONFIG_TEMPLATES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src", "mimose", "data", "config_templates"
)

MODALITY_COLS = ["Fl", "T1", "T1c", "T2"]

# Per-fold result workbooks (written by mimose.testing.pipeline) have no
# Fl/T1/T1c/T2 presence columns -- they list one row per modality
# combination in this fixed order (mimose's _string_to_order), followed by
# a trailing "Average" row. This mirrors the legacy sheet's row order, so
# the modality key is inferred from row position rather than read from a
# column.
MODALITY_KEY_ORDER = [
    (True, False, False, False),  # Fl
    (False, True, False, False),  # T1
    (False, False, True, False),  # T1c
    (False, False, False, True),  # T2
    (True, True, False, False),  # Fl+T1
    (True, False, True, False),  # Fl+T1c
    (True, False, False, True),  # Fl+T2
    (False, True, True, False),  # T1+T1c
    (False, True, False, True),  # T1+T2
    (False, False, True, True),  # T1c+T2
    (True, True, True, False),  # Fl+T1+T1c
    (True, True, False, True),  # Fl+T1+T2
    (True, False, True, True),  # Fl+T1c+T2
    (False, True, True, True),  # T1+T1c+T2
    (True, True, True, True),  # Fl+T1+T1c+T2
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def normalize(name: str) -> str:
    """Lowercase and strip every non-alphanumeric character."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def to_float(value):
    """Coerce a legacy/seed cell to float, returning None when not numeric."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def modality_key(row) -> tuple:
    """Boolean tuple of modality presence from the first four cells."""
    return tuple(bool(row[i]) and str(row[i]).strip().lower() == "x" for i in range(4))


# --------------------------------------------------------------------------- #
# Legacy parsing
# --------------------------------------------------------------------------- #
def parse_legacy(path):
    """Return legacy[dataset][region][modality_key][model_norm] = dice value.

    Also returns the ordered list of legacy display names keyed by normalized
    name so tabs can use the canonical display spelling.
    """
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[wb.sheetnames[0]]
    rows = [list(r) for r in ws.iter_rows(values_only=True)]

    legacy = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )  # ds -> region -> modkey -> modelnorm -> value
    display_names = {}  # model_norm -> display name

    # Locate each "Dataset: ..." marker row.
    marker_rows = []
    for i, row in enumerate(rows):
        for cell in row:
            if isinstance(cell, str) and cell.startswith("Dataset:"):
                ds = cell.split(":", 1)[1].strip()
                marker_rows.append((i, ds))
                break

    for marker_idx, ds in marker_rows:
        region_row = rows[marker_idx + 1]
        model_row = rows[marker_idx + 2]

        # Determine the region block that each column belongs to.
        col_region = {}
        current = None
        for c, cell in enumerate(region_row):
            if isinstance(cell, str) and cell.strip() in LEGACY_REGION_LABELS:
                current = LEGACY_REGION_LABELS[cell.strip()]
            if current is not None and c >= 4:
                col_region[c] = current

        # Column -> model (normalized + display).
        col_model = {}
        for c, cell in enumerate(model_row):
            if c >= 4 and isinstance(cell, str) and cell.strip():
                norm = normalize(cell)
                col_model[c] = norm
                display_names.setdefault(norm, cell.strip())

        # Data rows until the "Average" row.
        r = marker_idx + 3
        while r < len(rows):
            row = rows[r]
            if row and isinstance(row[0], str) and row[0].strip().lower() == "average":
                break
            if not any(row[i] for i in range(4)):
                r += 1
                continue
            mkey = modality_key(row)
            for c, model_norm in col_model.items():
                region = col_region.get(c)
                if region is None:
                    continue
                legacy[ds][region][mkey][model_norm] = to_float(row[c])
            r += 1

    return legacy, display_names


# --------------------------------------------------------------------------- #
# Seed parsing
# --------------------------------------------------------------------------- #
def parse_result_file(path):
    """Return {modality_key: {'WT':.., 'TC':.., 'ET':.., 'WT_hd95':..}} for a per-fold result file."""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[wb.sheetnames[0]]
    rows = [list(r) for r in ws.iter_rows(values_only=True)]

    # Find header row (the one containing 'WT' and 'TC').
    header_idx = None
    for i, row in enumerate(rows):
        vals = [str(c).strip() if c is not None else "" for c in row]
        if "WT" in vals and "TC" in vals:
            header_idx = i
            break
    if header_idx is None:
        return {}

    header = [str(c).strip() if c is not None else "" for c in rows[header_idx]]
    col_of = {name: idx for idx, name in enumerate(header)}

    wanted = []
    for region in REGIONS:
        wanted.append(region)
        wanted.append(f"{region}_hd95")

    # Data rows follow MODALITY_KEY_ORDER, one row per combination, then a
    # trailing "Average" row that we don't need (mean/std are recomputed
    # across folds below).
    data_rows = rows[header_idx + 1 : header_idx + 1 + len(MODALITY_KEY_ORDER)]

    out = {}
    for mkey, row in zip(MODALITY_KEY_ORDER, data_rows):
        entry = {}
        for w in wanted:
            idx = col_of.get(w)
            entry[w] = to_float(row[idx]) if idx is not None else None
        out[mkey] = entry
    return out


def fold_label(filename):
    """Extract 'fold1' from 'results_fold1.xlsx'."""
    m = re.search(r"fold(\d+)", filename)
    return f"fold{m.group(1)}" if m else filename


def discover(results_dir):
    """Scan results_dir for <model>_<dsnum> directories with per-fold workbooks.

    Returns models[model_norm][dataset_label] = {fold_label: fold_data_dict}.
    """
    models = defaultdict(dict)
    entries = [
        entry for entry in sorted(os.listdir(results_dir))
        if os.path.isdir(os.path.join(results_dir, entry))
    ]
    for entry in track(entries, description="Scanning result directories"):
        full = os.path.join(results_dir, entry)
        m = re.match(r"^(.*)_(\d+)$", entry)
        if not m:
            continue
        model_raw, dsnum = m.group(1), m.group(2)
        if dsnum not in DATASETS:
            continue
        dataset = DATASETS[dsnum]
        fold_files = sorted(
            glob.glob(os.path.join(full, "results_fold*.xlsx")),
            key=lambda p: int(re.search(r"fold(\d+)", p).group(1)),
        )
        if not fold_files:
            continue
        folds = {}
        for ff in fold_files:
            folds[fold_label(os.path.basename(ff))] = parse_result_file(ff)
        models[normalize(model_raw)][dataset] = folds
    return models


# --------------------------------------------------------------------------- #
# Workbook writing
# --------------------------------------------------------------------------- #
HEADER_FILL = PatternFill("solid", fgColor="4472C4")
SUBHEADER_FILL = PatternFill("solid", fgColor="D9E1F2")
TITLE_FILL = PatternFill("solid", fgColor="1F3864")
ERROR_FILL = PatternFill("solid", fgColor="FCE4D6")
WHITE = Font(color="FFFFFF", bold=True)
BOLD = Font(bold=True)
CENTER = Alignment(horizontal="center", vertical="center")
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def modality_order(legacy, dataset):
    """Ordered list of modality keys as they appear in the legacy sheet."""
    # Use whichever region has data; ordering is consistent across regions.
    for region in REGIONS:
        block = legacy.get(dataset, {}).get(region, {})
        if block:
            return list(block.keys())
    return []


def write_table(ws, start_row, dataset, legacy, folds_by_ds, display_name):
    """Write one dataset table starting at start_row. Return next free row."""
    folds = folds_by_ds.get(dataset, {})
    fold_labels = sorted(folds.keys(), key=lambda s: int(re.search(r"\d+", s).group()))

    # Column plan: for each metric (Dice, HD95) and each region, a block of
    #   Legacy | <fold labels...> | Mean | Std | Error
    per_region = ["Legacy"] + fold_labels + ["Mean", "Std", "Error"]
    metrics = [("Dice", ""), ("HD95", "_hd95")]

    # Title row.
    ws.cell(row=start_row, column=1, value=dataset).font = Font(color="FFFFFF", bold=True, size=12)
    title_end = 4 + len(metrics) * len(REGIONS) * len(per_region)
    ws.merge_cells(start_row=start_row, start_column=1, end_row=start_row, end_column=title_end)
    for c in range(1, title_end + 1):
        ws.cell(row=start_row, column=c).fill = TITLE_FILL
    ws.cell(row=start_row, column=1).alignment = Alignment(horizontal="left", vertical="center")

    grp_row = start_row + 1  # metric+region group header
    sub_row = start_row + 2  # per-column labels

    # Modality group header.
    ws.merge_cells(start_row=grp_row, start_column=1, end_row=grp_row, end_column=4)
    mc = ws.cell(row=grp_row, column=1, value="Modalities")
    mc.font, mc.fill, mc.alignment = WHITE, HEADER_FILL, CENTER
    for i, name in enumerate(MODALITY_COLS):
        c = ws.cell(row=sub_row, column=1 + i, value=name)
        c.font, c.fill, c.alignment, c.border = BOLD, SUBHEADER_FILL, CENTER, BORDER

    col = 5
    block_spans = []  # (start_col, region label, per_region list) for error highlighting
    for metric_name, _suffix in metrics:
        for region in REGIONS:
            span = len(per_region)
            ws.merge_cells(start_row=grp_row, start_column=col, end_row=grp_row, end_column=col + span - 1)
            gc = ws.cell(row=grp_row, column=col, value=f"{metric_name} · {region}")
            gc.font, gc.fill, gc.alignment = WHITE, HEADER_FILL, CENTER
            for j, label in enumerate(per_region):
                c = ws.cell(row=sub_row, column=col + j, value=label)
                c.font, c.alignment, c.border = BOLD, CENTER, BORDER
                c.fill = ERROR_FILL if label == "Error" else SUBHEADER_FILL
            block_spans.append((col, metric_name, region))
            col += span

    # Data rows.
    data_start = sub_row + 1
    mkeys = modality_order(legacy, dataset)
    for r_off, mkey in enumerate(mkeys):
        row = data_start + r_off
        for i, present in enumerate(mkey):
            c = ws.cell(row=row, column=1 + i, value="x" if present else None)
            c.alignment, c.border = CENTER, BORDER

        col = 5
        for metric_name, suffix in metrics:
            for region in REGIONS:
                legacy_val = None
                if metric_name == "Dice":
                    legacy_val = legacy.get(dataset, {}).get(region, {}).get(mkey, {}).get(
                        normalize(display_name)
                    )
                fold_vals = []
                for fl in fold_labels:
                    entry = folds.get(fl, {}).get(mkey, {})
                    fold_vals.append(entry.get(region + suffix))

                present_vals = [v for v in fold_vals if v is not None]
                mean = statistics.fmean(present_vals) if present_vals else None
                std = statistics.pstdev(present_vals) if len(present_vals) > 1 else (
                    0.0 if len(present_vals) == 1 else None
                )
                error = (legacy_val - mean) if (legacy_val is not None and mean is not None) else None

                values = [legacy_val] + fold_vals + [mean, std, error]
                for j, v in enumerate(values):
                    c = ws.cell(row=row, column=col + j)
                    if v is not None:
                        c.value = round(v, 4)
                    c.alignment, c.border = CENTER, BORDER
                    if per_region[j] == "Error" and v is not None:
                        c.fill = ERROR_FILL
                col += len(per_region)

    end_row = data_start + len(mkeys)
    return end_row + 2  # leave a blank spacer row


def safe_sheet_title(name):
    title = re.sub(r"[:\\/?*\[\]]", "-", name)
    return title[:31]


# --------------------------------------------------------------------------- #
# Completion report
# --------------------------------------------------------------------------- #
def expected_models(dsnum):
    """Normalized model names that have a `<model>_<dsnum>.yaml` config template."""
    pattern = os.path.join(CONFIG_TEMPLATES_DIR, f"*_{dsnum}.yaml")
    names = []
    suffix = f"_{dsnum}.yaml"
    for path in glob.glob(pattern):
        base = os.path.basename(path)[: -len(suffix)]
        names.append(normalize(base))
    return sorted(set(names))


def completion_report(models):
    """Return {dsnum: {'expected': [...], 'found': int, 'total': int, 'missing': [...]}}."""
    report = {}
    for dsnum, dataset in DATASETS.items():
        expected = expected_models(dsnum)
        total = len(expected) * len(EXPECTED_FOLDS)
        found = 0
        missing = []
        for model_norm in expected:
            folds = models.get(model_norm, {}).get(dataset, {})
            have = sorted(set(folds.keys()) & set(EXPECTED_FOLDS))
            found += len(have)
            for fold in EXPECTED_FOLDS:
                if fold not in have:
                    missing.append((model_norm, fold))
        report[dsnum] = {"expected": expected, "found": found, "total": total, "missing": missing}
    return report


def write_summary_sheet(wb, report, display_names, models):
    ws = wb.create_sheet(title="Summary", index=0)
    ws.sheet_view.showGridLines = False
    ws.column_dimensions["A"].width = 28
    for col in "BCDE":
        ws.column_dimensions[col].width = 12

    row = 1
    for dsnum, dataset in DATASETS.items():
        info = report[dsnum]
        pct = 100 * info["found"] / info["total"] if info["total"] else 0.0

        ws.cell(row=row, column=1, value=dataset).font = Font(bold=True, size=12)
        row += 1
        ws.cell(row=row, column=1, value="Completion:")
        ws.cell(row=row, column=2, value=f"{info['found']}/{info['total']} ({pct:.1f}%)").font = BOLD
        row += 2

        headers = ["Model"] + EXPECTED_FOLDS
        for c, h in enumerate(headers, start=1):
            cell = ws.cell(row=row, column=c, value=h)
            cell.font, cell.fill, cell.border = WHITE, HEADER_FILL, BORDER
        row += 1

        for model_norm in info["expected"]:
            display = display_names.get(model_norm, model_norm)
            folds = models.get(model_norm, {}).get(dataset, {})
            ws.cell(row=row, column=1, value=display).border = BORDER
            for c, fold in enumerate(EXPECTED_FOLDS, start=2):
                present = fold in folds
                cell = ws.cell(row=row, column=c, value="x" if present else "")
                cell.alignment, cell.border = CENTER, BORDER
                if not present:
                    cell.fill = ERROR_FILL
            row += 1

        row += 2

    return ws


def main():
    legacy_path = os.path.join(RESULTS_DIR, LEGACY_FILE)
    legacy, display_names = parse_legacy(legacy_path)
    models = discover(RESULTS_DIR)

    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    report = completion_report(models)
    write_summary_sheet(wb, report, display_names, models)

    # One tab per available model, sorted by display name.
    available = sorted(models.keys(), key=lambda n: display_names.get(n, n).lower())
    if not available:
        print("No available models found in", RESULTS_DIR)
        return

    for model_norm in track(available, description="Writing model tabs"):
        display = display_names.get(model_norm, model_norm)
        ws = wb.create_sheet(title=safe_sheet_title(display))
        ws.sheet_view.showGridLines = False
        folds_by_ds = models[model_norm]

        next_row = 1
        for dataset in DATASETS.values():
            next_row = write_table(ws, next_row, dataset, legacy, folds_by_ds, display)

        # Reasonable column widths.
        ws.column_dimensions["A"].width = 5
        for i in range(2, 5):
            ws.column_dimensions[get_column_letter(i)].width = 5
        for i in range(5, ws.max_column + 1):
            ws.column_dimensions[get_column_letter(i)].width = 9
        ws.freeze_panes = "E4"

    out_path = os.path.join(RESULTS_DIR, OUTPUT_FILE)
    wb.save(out_path)
    print(f"Wrote {out_path}")
    print(f"Tabs ({len(available)}): {', '.join(display_names.get(m, m) for m in available)}")
    for model_norm in available:
        for ds, folds in models[model_norm].items():
            print(f"  {display_names.get(model_norm, model_norm)} / {ds}: folds = {', '.join(sorted(folds))}")

    print()
    for dsnum, dataset in DATASETS.items():
        info = report[dsnum]
        pct = 100 * info["found"] / info["total"] if info["total"] else 0.0
        print(f"{dataset} completion: {info['found']}/{info['total']} ({pct:.1f}%)")
        for model_norm, fold in info["missing"]:
            print(f"  missing: {display_names.get(model_norm, model_norm)} / {fold}")


if __name__ == "__main__":
    main()
