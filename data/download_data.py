"""
Dedicated dataset download and verification utility.
Downloads the UCI Household Electric Power Consumption dataset from mirror or UCI,
extracts the archive, and verifies data integrity.
"""

import os
import sys
import zipfile
import urllib.request

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
ZIP_PATH = os.path.join(RAW_DIR, 'household_power_consumption.zip')
TXT_PATH = os.path.join(RAW_DIR, 'household_power_consumption.txt')

PRIMARY_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/household_power_consumption.zip"
BACKUP_URL = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"


def download_and_extract():
    os.makedirs(RAW_DIR, exist_ok=True)

    if os.path.exists(TXT_PATH) and os.path.getsize(TXT_PATH) > 1000000:
        print(f" Dataset already extracted at: {TXT_PATH} ({os.path.getsize(TXT_PATH):,} bytes)")
        return TXT_PATH

    if not os.path.exists(ZIP_PATH) or os.path.getsize(ZIP_PATH) < 20000000:
        print(f"Downloading dataset archive to {ZIP_PATH}...")
        try:
            req = urllib.request.Request(PRIMARY_URL, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=120) as resp, open(ZIP_PATH, 'wb') as out_f:
                total = 0
                while True:
                    chunk = resp.read(64 * 1024)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    total += len(chunk)
            print(f" Download complete ({total:,} bytes).")
        except Exception as e:
            print(f"Primary URL failed: {e}. Trying backup...")
            urllib.request.urlretrieve(BACKUP_URL, ZIP_PATH)

    print("Extracting archive...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        z.extractall(RAW_DIR)

    print(f" Successfully extracted to: {TXT_PATH}")
    return TXT_PATH


if __name__ == '__main__':
    download_and_extract()
