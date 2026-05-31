import yfinance as yf
import pandas as pd
import numpy as np
import pickle
import os
import time
import warnings

# Mengabaikan warning agar terminal bersih
warnings.filterwarnings('ignore')

# --- KONFIGURASI ---
START_DATE = '2020-01-01'
END_DATE = '2026-05-29'
SAVE_DIR = './dataset/IDX_ALL' # Folder terpisah untuk data full
TICKER_FILE = 'ihsg_all.csv'
CHUNK_SIZE = 50  # Batas aman request per batch
SLEEP_TIME = 2   # Jeda 2 detik antar batch agar tidak di-banned YF

def compute_rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=window-1, adjust=False).mean()
    ema_down = down.ewm(com=window-1, adjust=False).mean()
    rs = ema_up / ema_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def fetch_and_process_data():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    try:
        df_tickers = pd.read_csv(TICKER_FILE)
        ticker_list = df_tickers['Ticker'].dropna().unique().tolist()
    except FileNotFoundError:
        print(f"[ERROR] File {TICKER_FILE} tidak ditemukan.")
        return

    print(f"Memulai proses unduh {len(ticker_list)} saham dengan sistem Batching...")
    
    # 1. Proses Download Bertahap (Chunking)
    all_raw_data = pd.DataFrame()
    
    for i in range(0, len(ticker_list), CHUNK_SIZE):
        chunk = ticker_list[i:i + CHUNK_SIZE]
        print(f"Mengunduh Batch {i//CHUNK_SIZE + 1} ({len(chunk)} saham)...")
        
        try:
            # Unduh per batch
            batch_data = yf.download(chunk, start=START_DATE, end=END_DATE, group_by='ticker', auto_adjust=False, progress=False)
            
            # Gabungkan ke dataframe utama
            if all_raw_data.empty:
                all_raw_data = batch_data
            else:
                all_raw_data = pd.concat([all_raw_data, batch_data], axis=1)
                
        except Exception as e:
            print(f"[ERROR] Batch {i//CHUNK_SIZE + 1} gagal diunduh: {e}")
            
        # Jeda keamanan server
        time.sleep(SLEEP_TIME)

    # 2. Validasi Data Saham
    valid_tickers = []
    # YF mengembalikan MultiIndex columns jika >1 saham, atau single level jika 1 saham
    # Karena kita pasti >1 saham, kita ekstrak level 0
    available_tickers = all_raw_data.columns.levels[0] if isinstance(all_raw_data.columns, pd.MultiIndex) else []
    
    for ticker in ticker_list:
        if ticker in available_tickers:
            if not all_raw_data[ticker].dropna(how='all').empty:
                valid_tickers.append(ticker)

    num_stocks = len(valid_tickers)
    print(f"\n[INFO] Berhasil mengekstrak {num_stocks} saham valid dari total request.")
    
    if num_stocks == 0:
        print("[ERROR] Tidak ada data valid.")
        return

    # 3. Penyeragaman Kalender Bursa (Cari saham dengan umur terpanjang/data terbanyak)
    longest_ticker = max(valid_tickers, key=lambda t: len(all_raw_data[t].dropna(how='all')))
    reference_df = all_raw_data[longest_ticker].dropna(how='all')
    print(f"\n[INFO] Menggunakan {longest_ticker} sebagai referensi kalender.")
    dates = reference_df.index
    num_days = len(dates)
    num_features = 6  # OHLCV + RSI

    print(f"Menyusun Tensor Matriks (Hari: {num_days}, Saham: {num_stocks}, Fitur: {num_features})...")

    # Alokasi Memory (float32 agar hemat RAM & cocok untuk PyTorch)
    eod_data = np.zeros((num_days, num_stocks, num_features), dtype=np.float32)
    price_data = np.zeros((num_days, num_stocks), dtype=np.float32)
    mask_data = np.ones((num_days, num_stocks), dtype=np.float32)
    gt_data = np.zeros((num_days, num_stocks), dtype=np.float32)

    # 4. Ekstraksi Fitur & Matriks
    for s_idx, ticker in enumerate(valid_tickers):
        df = all_raw_data[ticker].copy()
        df = df.reindex(dates)

        # Imputasi data bolong
        df.ffill(inplace=True)
        df.bfill(inplace=True)

        df['RSI'] = compute_rsi(df['Close'], window=14)
        df['RSI'].bfill(inplace=True) 

        open_p = df['Open'].values
        high_p = df['High'].values
        low_p = df['Low'].values
        close_p = df['Close'].values
        volume = df['Volume'].values
        rsi_p = df['RSI'].values

        for t in range(num_days):
            eod_data[t, s_idx, 0] = open_p[t]
            eod_data[t, s_idx, 1] = high_p[t]
            eod_data[t, s_idx, 2] = low_p[t]
            eod_data[t, s_idx, 3] = close_p[t]
            eod_data[t, s_idx, 4] = volume[t]
            eod_data[t, s_idx, 5] = rsi_p[t]

            price_data[t, s_idx] = close_p[t]

            # Mask 0 untuk hari tanpa transaksi (saham gocap/suspend)
            if volume[t] == 0 or np.isnan(volume[t]):
                mask_data[t, s_idx] = 0.0

        # Ground Truth
        for t in range(num_days - 1):
            if close_p[t] != 0 and not np.isnan(close_p[t]):
                gt_data[t, s_idx] = (close_p[t+1] - close_p[t]) / close_p[t]
            else:
                gt_data[t, s_idx] = 0.0
        gt_data[-1, s_idx] = 0.0 

    # 5. Normalisasi
    print("Mengeksekusi Z-Score Normalization...")
    for s_idx in range(num_stocks):
        for f_idx in range(num_features):
            mean_val = np.mean(eod_data[:, s_idx, f_idx])
            std_val = np.std(eod_data[:, s_idx, f_idx])
            if std_val != 0 and not np.isnan(std_val):
                eod_data[:, s_idx, f_idx] = (eod_data[:, s_idx, f_idx] - mean_val) / std_val
            else:
                eod_data[:, s_idx, f_idx] = 0.0

    # 6. Ekspor Data
    with open(os.path.join(SAVE_DIR, 'eod_data.pkl'), 'wb') as f:
        pickle.dump(eod_data, f)
    with open(os.path.join(SAVE_DIR, 'mask_data.pkl'), 'wb') as f:
        pickle.dump(mask_data, f)
    with open(os.path.join(SAVE_DIR, 'gt_data.pkl'), 'wb') as f:
        pickle.dump(gt_data, f)
    with open(os.path.join(SAVE_DIR, 'price_data.pkl'), 'wb') as f:
        pickle.dump(price_data, f)

    print(f"\n[SUCCESS] Pipeline selesai! Bentuk Matriks EOD : {eod_data.shape}")

if __name__ == "__main__":
    fetch_and_process_data()