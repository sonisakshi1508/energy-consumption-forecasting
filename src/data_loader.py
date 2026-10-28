"""
Data loading and preprocessing module for the UCI Individual Household
Electric Power Consumption dataset.
"""

import os
import zipfile
import urllib.request
import pandas as pd
import numpy as np

# Default file paths
RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed')
RAW_ZIP_PATH = os.path.join(RAW_DIR, 'household_power_consumption.zip')
RAW_TXT_PATH = os.path.join(RAW_DIR, 'household_power_consumption.txt')
PROCESSED_CSV_PATH = os.path.join(PROCESSED_DIR, 'hourly_energy_consumption.csv')

PRIMARY_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/household_power_consumption.zip"
BACKUP_URL = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"


def download_dataset(destination_zip: str = RAW_ZIP_PATH, extract_to: str = RAW_DIR) -> str:
    """
    Downloads and extracts the UCI Household Electric Power Consumption dataset
    if it is not already present locally.
    """
    os.makedirs(extract_to, exist_ok=True)
    target_txt = os.path.join(extract_to, 'household_power_consumption.txt')

    if os.path.exists(target_txt):
        print(f"[DataLoader] Raw dataset already exists at: {target_txt}")
        return target_txt

    urls = [PRIMARY_URL, BACKUP_URL]
    downloaded = False

    for url in urls:
        try:
            print(f"[DataLoader] Downloading dataset from {url}...")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=60) as resp, open(destination_zip, 'wb') as out_file:
                chunk_size = 64 * 1024
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
            print(f"[DataLoader] Successfully downloaded to {destination_zip}")
            downloaded = True
            break
        except Exception as e:
            print(f"[DataLoader] Failed to download from {url}: {e}")

    if not downloaded and not os.path.exists(target_txt):
        raise RuntimeError("Unable to download the dataset from any source. Please verify your internet connection.")

    if os.path.exists(destination_zip):
        print("[DataLoader] Extracting zip file...")
        with zipfile.ZipFile(destination_zip, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"[DataLoader] Extraction complete. Raw file at: {target_txt}")

    return target_txt


def preprocess_raw_data(raw_txt_path: str = RAW_TXT_PATH, output_csv_path: str = PROCESSED_CSV_PATH) -> pd.DataFrame:
    """
    Loads raw minute-level data, handles missing values, converts timestamps,
    and resamples to hourly observations.

    Parameters:
        raw_txt_path: Path to the unzipped household_power_consumption.txt file.
        output_csv_path: Path to save the processed hourly CSV.

    Returns:
        pd.DataFrame: Resampled hourly dataset indexed by Datetime.
    """
    if not os.path.exists(raw_txt_path):
        raw_txt_path = download_dataset(extract_to=os.path.dirname(raw_txt_path))

    print("[DataLoader] Reading raw dataset (semicolon-delimited, missing values as '?')...")
    # Read raw data
    df = pd.read_csv(
        raw_txt_path,
        sep=';',
        na_values=['?', ' ?'],
        dtype={
            'Global_active_power': 'float64',
            'Global_reactive_power': 'float64',
            'Voltage': 'float64',
            'Global_intensity': 'float64',
            'Sub_metering_1': 'float64',
            'Sub_metering_2': 'float64',
            'Sub_metering_3': 'float64'
        },
        low_memory=False
    )

    print(f"[DataLoader] Loaded {len(df):,} raw records. Parsing dates and handling missing values...")

    # Combine Date and Time columns into a single datetime column
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
    df = df.dropna(subset=['Datetime'])
    df = df.set_index('Datetime')
    df = df.drop(columns=['Date', 'Time'])

    # Sort index chronologically
    df = df.sort_index()

    # Resample minute-level observations into hourly observations by calculating the mean
    print("[DataLoader] Resampling minute-level observations to hourly averages...")
    hourly_df = df.resample('1h').mean()

    # Interpolate any hourly missing values created during resampling or missing intervals
    # Linear interpolation for gaps, followed by forward/backward fill for boundary values
    hourly_df = hourly_df.interpolate(method='time')
    hourly_df = hourly_df.ffill().bfill()

    # Save to processed directory for instant loading in future runs
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    hourly_df.to_csv(output_csv_path)
    print(f"[DataLoader] Processed hourly dataset saved to: {output_csv_path} ({len(hourly_df):,} hourly records).")

    return hourly_df


def load_hourly_data(csv_path: str = PROCESSED_CSV_PATH, force_reprocess: bool = False) -> pd.DataFrame:
    """
    Convenience function to load the processed hourly dataset.
    If the processed file doesn't exist or force_reprocess is True,
    it downloads (if necessary) and processes raw data.
    """
    if os.path.exists(csv_path) and not force_reprocess:
        print(f"[DataLoader] Loading cached hourly dataset from: {csv_path}")
        df = pd.read_csv(csv_path, parse_dates=['Datetime'], index_col='Datetime')
        # Ensure uniform hourly frequency
        if df.index.freq is None:
            df = df.asfreq('1h')
            df = df.interpolate(method='time').ffill().bfill()
        return df

    return preprocess_raw_data(output_csv_path=csv_path)


if __name__ == '__main__':
    df_hourly = load_hourly_data()
    print("[DataLoader] Dataset Preview:")
    print(df_hourly.head())
    print("\n[DataLoader] Summary Statistics:")
    print(df_hourly[['Global_active_power', 'Voltage', 'Global_intensity']].describe())
