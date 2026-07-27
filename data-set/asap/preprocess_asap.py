#!/usr/bin/env python
import argparse
import codecs
import sys

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i", "--input-file", dest="input_file", required=True, help="Input TSV file"
)
args = parser.parse_args()


def collect_dataset(input_file):
    dataset = dict()
    with open(input_file, encoding="utf-8") as f:
        dataset["header"] = next(f)  # First line is the header
        for line in f:
            parts = line.strip().split("\t")
            assert len(parts) >= 6, f"ERROR: {line}"
            dataset[parts[0]] = line
    return dataset


def extract_based_on_ids(dataset, id_file):
    lines = []
    with open(id_file, encoding="utf-8") as f:
        for line in f:
            id = line.strip()
            if id in dataset:
                lines.append(dataset[id])
            else:
                print(f"ERROR: Invalid ID {id} in {id_file}", file=sys.stderr)
    return lines


def create_dataset(lines, output_fname, dataset):
    with open(output_fname, "w", encoding="utf-8") as f_write:
        f_write.write(dataset["header"])
        for line in lines:
            f_write.write(line)


dataset = collect_dataset(args.input_file)

for fold_idx in range(5):
    for dataset_type in ["dev", "test", "train"]:
        id_file = f"fold_{fold_idx}/{dataset_type}_ids.txt"
        output_file = f"fold_{fold_idx}/{dataset_type}.tsv"
        lines = extract_based_on_ids(dataset, id_file)
        create_dataset(lines, output_file, dataset)
