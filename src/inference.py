import torch
import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from model import StockMixer
import warnings

warnings.filterwarnings('ignore')

# --- KONFIGURASI ---
MARKET_NAME = 'IDX_ALL'
STOCK_NUM = 958
FEA_NUM = 6
LOOKBACK = 16
MODEL_PATH = f'models/{MARKET_NAME}_best.pth'  
DATA_DIR = '../dataset/IDX_ALL'
TICKER_FILE = '../ihsg_all.csv'

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
    model = StockMixer(stocks=STOCK_NUM, time_steps=LOOKBACK, channels=FEA_NUM, market=20, scale=1.0).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() 

    print("2. Menyiapkan Data Pasar...")
    with open(os.path.join(DATA_DIR, "eod_data.pkl"), "rb") as f:
        eod_data = pickle.load(f).transpose(1, 0, 2)

    latest_data = eod_data[:, -LOOKBACK:, :] 
    latest_tensor = torch.tensor(latest_data, dtype=torch.float32).to(device)

    print("3. Analisis Prediktif AI...")
    with torch.no_grad(): 
        predictions = model(latest_tensor).cpu().numpy().flatten()

    tickers = load_tickers()[:STOCK_NUM] 
    df_results = pd.DataFrame({'Ticker': tickers, 'Score': predictions})
    
    # Kita ambil Top 300 dan Bottom 300 (total 600) untuk di cek YFinance agar tidak terlalu berat
    df_results = df_results.sort_values(by='Score', ascending=False)
    target_tickers = df_results.head(300)['Ticker'].tolist() + df_results.tail(300)['Ticker'].tolist()

    print(f"4. [RISK MANAGEMENT] Memeriksa Likuiditas di Bursa secara massal...")
    # Unduh massal untuk mempercepat waktu
    yf_data = yf.download(target_tickers, period="5d", group_by='ticker', progress=False)

    valid_stocks = []
    
    for ticker in target_tickers:
        try:
            df_ticker = yf_data[ticker] if len(target_tickers) > 1 else yf_data
            if df_ticker.empty or df_ticker['Close'].isna().all(): continue

            latest_close = float(df_ticker['Close'].iloc[-1])
            adtv = float((df_ticker['Close'] * df_ticker['Volume']).mean())
            adtv_billion = adtv / 1_000_000_000 

            if latest_close >= MIN_PRICE and adtv_billion >= MIN_ADTV_BILLION:
                score = df_results.loc[df_results['Ticker'] == ticker, 'Score'].values[0]
                valid_stocks.append({
                    'Ticker': ticker,
                    'AI_Score': score,
                    'Harga_Terakhir': latest_close,
                    'Trx_Harian_Miliar': round(adtv_billion, 2)
                })
        except:
            continue

    df_valid = pd.DataFrame(valid_stocks).sort_values(by='AI_Score', ascending=False).reset_index(drop=True)
    
    # Simpan SELURUH data valid ke CSV agar bisa dibaca di Excel
    csv_filename = '../rekomendasi_harian_lengkap.csv'
    df_valid.to_csv(csv_filename, index=False)
    print(f"\n[SUCCESS] Keseluruhan klasemen ({len(df_valid)} saham likuid) berhasil disimpan di: {csv_filename}")

    # Ekstrak Top 10 (Bullish) dan Bottom 10 (Bearish)
    top_10 = df_valid.head(10)
    bottom_10 = df_valid.tail(10)
    
    # --- VISUALISASI MATPLOTLIB: BULL VS BEAR ---
    fig, ax = plt.subplots(figsize=(15, 8))
    
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
    plt.tight_layout()
    plt.savefig('../top_bottom_prediction.png')
    plt.show()

if __name__ == "__main__":
    run_inference()