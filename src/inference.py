import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import pickle
import os
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from model import StockMixer
import warnings


warnings.filterwarnings('ignore')

# --- KONFIGURASI ---
MARKET_NAME = 'IDX_ALL'
STOCK_NUM = 958
FEA_NUM = 6
LOOKBACK = 16

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('--date', type=str, default=None)
_args, _ = _parser.parse_known_args()

today_str = _args.date if _args.date else datetime.datetime.now().strftime('%Y-%m-%d')

# Robust paths - use absolute paths to avoid working directory issues
base_dir = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(base_dir, '..', 'outputs', today_str)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(base_dir, 'models', f'{MARKET_NAME}_best.pth')  
DATA_DIR = os.path.join(base_dir, '..', 'dataset', MARKET_NAME)
TICKER_FILE = os.path.join(base_dir, '..', 'ihsg_all.csv')

# --- RISK MANAGEMENT ---
MIN_PRICE = 51                  # Hindari saham gocap
MIN_ADTV_BILLION = 1.0          # Transaksi harian minimal Rp 1 Miliar


def load_tickers():
    try:
        df_tickers = pd.read_csv(TICKER_FILE)
        return df_tickers['Ticker'].dropna().unique().tolist()
    except Exception as e:
        return [f"STOCK_{i}" for i in range(STOCK_NUM)] 

def run_inference():
    print("1. Memuat Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dynamically determine STOCK_NUM from the checkpoint to prevent shape mismatch
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    checkpoint_stocks = checkpoint['stock_mixer.dense1.weight'].shape[1]
    print(f"Model checkpoint stocks: {checkpoint_stocks}")
    
    model = StockMixer(stocks=checkpoint_stocks, time_steps=LOOKBACK, channels=FEA_NUM, market=20, scale=1.0).to(device)
    model.load_state_dict(checkpoint)
    model.eval() 

    print("2. Menyiapkan Data Pasar...")
    with open(os.path.join(DATA_DIR, "eod_data.pkl"), "rb") as f:
        eod_data = pickle.load(f).transpose(1, 0, 2)

    # Slice or pad dataset to match the model's stock dimension
    num_dataset_stocks = eod_data.shape[0]
    print(f"Dataset stocks: {num_dataset_stocks}")
    if num_dataset_stocks >= checkpoint_stocks:
        latest_data = eod_data[:checkpoint_stocks, -LOOKBACK:, :]
    else:
        pad_width = checkpoint_stocks - num_dataset_stocks
        latest_data = np.pad(eod_data[:, -LOOKBACK:, :], ((0, pad_width), (0, 0), (0, 0)), mode='constant')

    latest_tensor = torch.tensor(latest_data, dtype=torch.float32).to(device)

    print("3. Analisis Prediktif AI...")
    with torch.no_grad(): 
        predictions = model(latest_tensor).cpu().numpy().flatten()

    tickers = load_tickers()[:checkpoint_stocks] 
    df_results = pd.DataFrame({'Ticker': tickers, 'Score': predictions})
    
    # Kita ambil Top 300 dan Bottom 300 (total 600) untuk di cek YFinance agar tidak terlalu berat
    df_results = df_results.sort_values(by='Score', ascending=False)
    target_tickers = df_results.head(300)['Ticker'].tolist() + df_results.tail(300)['Ticker'].tolist()

    print(f"4. [RISK MANAGEMENT] Memeriksa Likuiditas di Bursa secara massal...")
    # Unduh massal untuk mempercepat waktu
    yf_data = yf.download(target_tickers, period="100d", group_by='ticker', progress=False)

    valid_stocks = []
    
    for ticker in target_tickers:
        try:
            df_ticker = yf_data[ticker] if len(target_tickers) > 1 else yf_data
            if df_ticker.empty or len(df_ticker) < 50: continue

            df_ticker['MA50'] = df_ticker['Close'].rolling(window=50).mean()
            latest_close = float(df_ticker['Close'].iloc[-1])
            ma50 = float(df_ticker['MA50'].iloc[-1])
            
            recent_5d = df_ticker.tail(5)
            adtv = float((recent_5d['Close'] * recent_5d['Volume']).mean())
            adtv_billion = adtv / 1_000_000_000 

            if latest_close >= MIN_PRICE and adtv_billion >= MIN_ADTV_BILLION:
                score = df_results.loc[df_results['Ticker'] == ticker, 'Score'].values[0]
                valid_stocks.append({
                    'Ticker': ticker,
                    'AI_Score': score,
                    'Harga': latest_close,
                    'MA50': round(ma50, 1),
                    'Trx_Miliar': round(adtv_billion, 2)
                })
        except Exception as e:
            continue


    df_valid = pd.DataFrame(valid_stocks).sort_values(by='AI_Score', ascending=False).reset_index(drop=True)
    
    csv_filename = os.path.join(OUTPUT_DIR, 'rekomendasi_lengkap_IDX_ALL.csv')
    df_valid.to_csv(csv_filename, index=False)
    print(f"\n[SUCCESS] Keseluruhan klasemen ({len(df_valid)} saham likuid) berhasil disimpan di: {csv_filename}")

    # Ekstrak Top 10 (Bullish) dan Bottom 10 (Bearish)
    top_10 = df_valid.head(10)
    bottom_10 = df_valid.tail(10)
    
    # --- VISUALISASI MATPLOTLIB: BULL VS BEAR ---
    fig, ax = plt.subplots(figsize=(15, 8))
    
    if len(df_valid) > 0:
        # Plot Bottom 10 (Merah - Negatif) - Diurutkan dari yang paling jelek
        bars_bottom = ax.bar(bottom_10['Ticker'], bottom_10['AI_Score'], color='#e74c3c', label='Strong Bearish (Avoid/Sell)')
        # Plot Top 10 (Hijau - Positif)
        bars_top = ax.bar(top_10['Ticker'], top_10['AI_Score'], color='#2ecc71', label='Strong Bullish (Buy)')
        
        ax.set_title('AI Prediction: Top 10 Bullish vs Top 10 Bearish (Liquidity Filtered)', fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('Relative Conviction Score', fontsize=12)
        ax.axhline(0, color='black', linewidth=1) # Garis batas 0
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.legend()

        # Tambahkan teks angka di bar hijau
        for bar in bars_top:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + (abs(yval)*0.02), f'{yval:.0f}', ha='center', va='bottom', fontsize=9)
        
        # Tambahkan teks angka di bar merah
        for bar in bars_bottom:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval - (abs(yval)*0.05), f'{yval:.0f}', ha='center', va='top', fontsize=9)

        plt.xticks(rotation=45)
    else:
        # Fallback: Plot pesan ketika tidak ada data
        ax.text(0.5, 0.5, 'No Valid Stocks\nPassing Liquidity Filter', 
                ha='center', va='center', fontsize=18, fontweight='bold',
                transform=ax.transAxes, color='gray')
        ax.set_title('AI Prediction: No Data Available', fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
    
    plt.tight_layout()
    png_filename = os.path.join(OUTPUT_DIR, 'visualisasi_IDX_ALL.png')
    plt.savefig(png_filename, dpi=150)
    print(f"[SUCCESS] Gambar Visualisasi disimpan di: {png_filename}")
    plt.close()


if __name__ == "__main__":
    run_inference()