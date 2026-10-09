"""Data cleaning pipeline for customer records."""
import csv
import sys
from pathlib import Path


def clean_records(input_csv: str, output_csv: str):
    print(f"Reading from {input_csv}...")
    cleaned = []
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            email = row.get("email", "").strip().lower()
            if "@" in email:
                row["email"] = email
                row["status"] = "verified"
                cleaned.append(row)

    print(f"Writing {len(cleaned)} cleaned records to {output_csv}...")
    if cleaned:
        with open(output_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cleaned[0].keys())
            writer.writeheader()
            writer.writerows(cleaned)


if __name__ == "__main__":
    clean_records("raw_users.csv", "cleaned_users.csv")
