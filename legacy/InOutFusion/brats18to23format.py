from argparse import ArgumentParser
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import shutil
import gzip
from tqdm import tqdm


NAME_MAP = {
    'seg.nii': 'seg.nii.gz',
    't1.nii': 't1n.nii.gz',
    't2.nii': 't2w.nii.gz',
    't1ce.nii': 't1c.nii.gz',
    'flair.nii': 't2f.nii.gz',
}


def save_compressed(source: Path, destination: Path) -> None:
    with open(source, 'rb') as f_in:
        with gzip.open(destination, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def process_one(task: tuple[Path, Path]) -> None:
    source, destination = task
    save_compressed(source, destination)


def build_tasks(input18: Path, output23: Path) -> list[tuple[Path, Path]]:
    tasks: list[tuple[Path, Path]] = []

    for grade in input18.iterdir():
        if not grade.is_dir():
            continue

        for sub18 in grade.iterdir():
            if not sub18.is_dir():
                continue

            sub23 = output23 / sub18.name
            sub23.mkdir(parents=True, exist_ok=True)

            for image in sub18.iterdir():
                if not image.is_file():
                    continue

                ext = image.name.split('_')[-1]
                if ext not in NAME_MAP:
                    raise RuntimeError(f'Invalid image: {image}')

                new_ext = NAME_MAP[ext]
                destination = sub23 / f'{sub23.name}-{new_ext}'
                tasks.append((image, destination))

    return tasks


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('--input18', type=Path, required=True)
    parser.add_argument('--output23', type=Path, required=True)
    parser.add_argument('--num_workers', type=int, default=32)
    args = parser.parse_args()

    if not args.input18.is_dir():
        raise RuntimeError(f'Invalid input18 directory: {args.input18}')

    if args.output23.exists():
        if args.output23.is_dir():
            answer = input(f'Output directory "{args.output23}" exists. Delete it? [y/N]: ').strip().lower()
            if answer in ('y', 'yes'):
                shutil.rmtree(args.output23)
            else:
                print('Aborting.')
                return
        else:
            answer = input(f'File "{args.output23}" exists. Delete it? [y/N]: ').strip().lower()
            if answer in ('y', 'yes'):
                args.output23.unlink()
            else:
                print('Aborting.')
                return

    args.output23.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(args.input18, args.output23)

    if args.num_workers <= 1:
        for task in tqdm(tasks, desc="Processing", unit="file"):
            process_one(task)
    else:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            list(tqdm(
                executor.map(process_one, tasks),
                total=len(tasks),
                desc="Processing",
                unit="file"
            ))


if __name__ == '__main__':
    main()