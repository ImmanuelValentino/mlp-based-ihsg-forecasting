import yfinance as yf
import pandas as pd
import numpy as np
import os
import pickle
import warnings

warnings.filterwarnings('ignore')

# Konfigurasi
TICKER_FILE = 'ihsg_all.csv' # Pastikan file ini ada di luar folder src/
OUTPUT_DIR = 'dataset/IDX_ALL_V2'
START_DATE = '2018-01-01'
END_DATE = '2026-06-11'

def build_dataset_v2(end_date=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    target_end_date = end_date if end_date else END_DATE
    print(f"Target End Date untuk V2: {target_end_date}")
    
    print(f"1. Membaca daftar saham dari {TICKER_FILE}...")
    df_tickers = pd.read_csv(TICKER_FILE)
    valid_tickers = df_tickers['Ticker'].dropna().unique().tolist()
    
    print(f"2. Mengunduh data historis untuk {len(valid_tickers)} saham...")
    all_raw_data = yf.download(valid_tickers, start=START_DATE, end=target_end_date, group_by='ticker', auto_adjust=True, progress=True)

    
    # Mencari kalender referensi dari saham paling tua
    longest_ticker = max(valid_tickers, key=lambda t: len(all_raw_data[t].dropna(how='all')) if t in all_raw_data else 0)
    master_dates = all_raw_data[longest_ticker].dropna(how='all').index
    print(f"3. Kalender Master menggunakan {longest_ticker} dengan {len(master_dates)} hari bursa.")

    all_eod_data, all_mask_data, all_gt_data = [], [], []

    print("4. Memproses indikator dan normalisasi untuk setiap saham...")
    for ticker in valid_tickers:
        if ticker not in all_raw_data:
            continue
            
        df = all_raw_data[ticker].reindex(master_dates)
        
        # --- FEATURE ENGINEERING ---
        # 1. RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI'] = df['RSI'].fillna(50)

        # 2. MACD
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = (ema12 - ema26).fillna(0)

        # 3. OBV
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

        # --- GROUND TRUTH (1 Hari Kedepan) ---
        df['Return'] = df['Close'].pct_change()
        gt_array = df['Return'].shift(-1).fillna(0).values

        # --- MASKING & Z-SCORE NORMALIZATION ---
        mask_array = (~df['Close'].isna()).astype(float).values
        
        # 8 Fitur V2
        features = df[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'OBV']]
        normalized_features = (features - features.mean()) / features.std()
        normalized_features = normalized_features.fillna(0).values

        all_eod_data.append(normalized_features)
        all_mask_data.append(mask_array)
        all_gt_data.append(gt_array)

    # Convert ke Tensor 3D Numpy
    eod_matrix = np.array(all_eod_data)
    mask_matrix = np.array(all_mask_data)
    gt_matrix = np.array(all_gt_data)

    print(f"5. Matriks berhasil dibuat! Shape: {eod_matrix.shape}")
    
    with open(os.path.join(OUTPUT_DIR, "eod_data.pkl"), "wb") as f:
        pickle.dump(eod_matrix, f)
    with open(os.path.join(OUTPUT_DIR, "mask_data.pkl"), "wb") as f:
        pickle.dump(mask_matrix, f)
    with open(os.path.join(OUTPUT_DIR, "gt_data.pkl"), "wb") as f:
        pickle.dump(gt_matrix, f)
        
    print(f"Penyimpanan selesai di {OUTPUT_DIR}")

if __name__ == '__main__':
    import argparse
    import datetime
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--end-date', type=str, default=None, help="End date for the dataset (YYYY-MM-DD)")
    args = parser.parse_args()
    
    # default to today's date if not passed
    target_date = args.end_date if args.end_date else datetime.datetime.now().strftime('%Y-%m-%d')
    build_dataset_v2(target_date)