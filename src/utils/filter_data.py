"""
filter_data.py

Временный скрипт - чтобы выбрать нужные колонки из датасета

"""

import sys
import argparse
import csv

def main():
    parser = argparse.ArgumentParser(description="Build correct labels file. That's a temporary script to be removed later.")
    parser.add_argument('--input', required=True, help='Path to CSV file')
    parser.add_argument('--output', required=True, help='Path to output .npz file')

    args = parser.parse_args()

    input_file = args.input
    output_file = args.output

    with open(input_file, mode='r', encoding='utf-8') as infile, \
            open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = ['text', 'reasoning_label_final']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            text = row.get('text', '')
            label = row.get('reasoning_label') or row.get('reasoning_label_old') or ''
            writer.writerow({'text': text, 'reasoning_label_final': label})

if __name__ == "__main__":
    main()