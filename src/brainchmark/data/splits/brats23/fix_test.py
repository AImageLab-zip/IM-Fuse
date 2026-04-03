from pathlib import Path
import csv
import sys


def csv_first_column_to_txt(input_csv: str, output_txt: str) -> None:
    with Path(input_csv).open("r", encoding="utf-8", newline="") as f_in:
        reader = csv.reader(f_in)
        next(reader, None)  # skip header

        with Path(output_txt).open("w", encoding="utf-8") as f_out:
            for row in reader:
                if len(row) >= 1:
                    f_out.write(row[0] + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py input.csv output.txt")
        sys.exit(1)

    csv_first_column_to_txt(sys.argv[1], sys.argv[2])