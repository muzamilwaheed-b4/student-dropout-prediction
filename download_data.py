"""
Dataset Downloader Helper
Downloads and extracts the UCI Student Dropout and Academic Success dataset.
"""
import urllib.request
import os
import zipfile

os.makedirs("data", exist_ok=True)

url = "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip"
zip_path = "data/dataset.zip"

print("Downloading UCI dataset...")
try:
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall("data/")
    for f in os.listdir("data/"):
        if f.endswith(".csv") and "data" not in f.lower():
            os.rename(f"data/{f}", "data/data.csv")
            print(f"Renamed {f} -> data/data.csv")
            break
    os.remove(zip_path)
    print("[OK] Dataset ready at data/data.csv")
except Exception as e:
    print(f"[ERROR] Download failed: {e}")
    print("\nManual download:")
    print("  Kaggle: https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention")
    print("  Save as: data/data.csv")
    print("  Note: Separator is semicolon (;)")
