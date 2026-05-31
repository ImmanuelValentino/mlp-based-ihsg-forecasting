import torch
import pickle
import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from model import StockMixer
import warnings

warnings.filterwarnings('ignore')

# --- KONFIGURASI V3 ---
MARKET_NAME = 'IDX_ALL_V3'      
FEA_NUM = 8                     
LOOKBACK = 16
MODEL_PATH = f'src/models/{MARKET_NAME}_best.pth'  
DATA_DIR = f'dataset/{MARKET_NAME}'
TICKER_FILE = 'ihsg_all.csv'

# --- RISK MANAGEMENT ---
MIN_PRICE = 51                  
MIN_ADTV_BILLION = 1.0          

today_str = datetime.datetime.now().strftime('%Y-%m-%d')
OUTPUT_DIR = f'outputs/{today_str}'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_tickers(stock_num):
    try:
        df_tickers = pd.read_csv(TICKER_FILE)
        return df_tickers['Ticker'].dropna().unique().tolist()
    except Exception:
        return [f"STOCK_{i}" for i in range(stock_num)] 

def run_inference_v2():
    print(f"[{today_str}] 1. Menyiapkan Data Pasar...")
    with open(os.path.join(DATA_DIR, "eod_data.pkl"), "rb") as f:
        eod_data = pickle.load(f)
        if eod_data.shape[0] > 1000:
            eod_data = eod_data.transpose(1, 0, 2)
            
    STOCK_NUM = eod_data.shape[0] 

    print(f"[{today_str}] 2. Memuat Model V3...")
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] File model {MODEL_PATH} tidak ditemukan.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StockMixer(stocks=STOCK_NUM, time_steps=LOOKBACK, channels=FEA_NUM, market=20, scale=1.0).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() 

    latest_data = eod_data[:, -LOOKBACK:, :] 
    latest_tensor = torch.tensor(latest_data, dtype=torch.float32).to(device)

    print(f"[{today_str}] 3. Analisis Prediktif AI (Swing 3 Hari)...")
    with torch.no_grad(): 
        predictions = model(latest_tensor).cpu().numpy().flatten()

    tickers = load_tickers(STOCK_NUM)[:STOCK_NUM] 
    df_results = pd.DataFrame({'Ticker': tickers, 'Score': predictions})
    df_results = df_results.sort_values(by='Score', ascending=False)
    
    target_tickers = df_results.head(300)['Ticker'].tolist() + df_results.tail(300)['Ticker'].tolist()

    print(f"[{today_str}] 4. [RISK MANAGEMENT] Memeriksa Bursa (Mencari Minimal Top 15)...")
    valid_stocks_bull = []
    valid_stocks_all = [] # Keranjang besar
    
    CHUNK_SIZE = 25 

    for i in range(0, len(target_tickers), CHUNK_SIZE):
        chunk = target_tickers[i:i + CHUNK_SIZE]
        print(f"   -> Inspeksi Batch {i//CHUNK_SIZE + 1} ({len(chunk)} saham)...")
        try:
            yf_data = yf.download(chunk, period="100d", group_by='ticker', progress=False)

            for ticker in chunk:
                try:
                    df_ticker = yf_data[ticker] if len(chunk) > 1 else yf_data
                    if df_ticker.empty or len(df_ticker) < 50: continue

                    df_ticker['MA50'] = df_ticker['Close'].rolling(window=50).mean()
                    latest_close = float(df_ticker['Close'].iloc[-1])
                    ma50 = float(df_ticker['MA50'].iloc[-1])
                    
                    recent_5d = df_ticker.tail(5)
                    adtv = float((recent_5d['Close'] * recent_5d['Volume']).mean())
                    adtv_billion = adtv / 1_000_000_000 

                    # Syarat Mutlak: Bukan Gocap & Likuid
                    if latest_close >= MIN_PRICE and adtv_billion >= MIN_ADTV_BILLION:
                        score = df_results.loc[df_results['Ticker'] == ticker, 'Score'].values[0]
                        stock_info = {
                            'Ticker': ticker,
                            'AI_Score': score,
                            'Harga': latest_close,
                            'MA50': round(ma50, 1),
                            'Trx_Miliar': round(adtv_billion, 2)
                        }

                        # Masukkan ke keranjang besar
                        valid_stocks_all.append(stock_info) 
                        
                        # Syarat tambahan khusus untuk Top Pick (Bullish): WAJIB Uptrend
                        if latest_close > ma50:
                            valid_stocks_bull.append(stock_info)
                except Exception:
                    continue
        except Exception as e:
            print(f"   [!] Gagal mengunduh batch: {e}")

    # Trik Anti-KeyError: Definisikan kolom secara eksplisit
    columns = ['Ticker', 'AI_Score', 'Harga', 'MA50', 'Trx_Miliar']
    
    df_valid_bull = pd.DataFrame(valid_stocks_bull, columns=columns).sort_values(by='AI_Score', ascending=False)
    df_valid_all = pd.DataFrame(valid_stocks_all, columns=columns).sort_values(by='AI_Score', ascending=False)
    
    if df_valid_all.empty:
        print("\n[WARNING] Tidak ada satupun saham yang lolos filter likuiditas hari ini.")
        return

    csv_filename = f'{OUTPUT_DIR}/rekomendasi_lengkap_{MARKET_NAME}.csv'
    df_valid_all.to_csv(csv_filename, index=False)
    print(f"\n[SUCCESS] File CSV disimpan di: {csv_filename}")

    # --- RENDER GAMBAR ---
    top_15_bull = df_valid_bull.head(15)
    top_15_bear = df_valid_all.tail(15) # Ambil 15 terburuk dari keranjang besar
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))

    # Kiri: Bullish
    bars_bull = axes[0].barh(top_15_bull['Ticker'][::-1], top_15_bull['AI_Score'][::-1], color='#2ecc71', edgecolor='black', linewidth=0.5)
    axes[0].set_title(f'🚀 Top {len(top_15_bull)} Strong Bullish (Uptrend Validated)', fontsize=16, fontweight='bold', pad=15)
    axes[0].grid(axis='x', linestyle='--', alpha=0.5)
    for bar in bars_bull:
        score_val = bar.get_width()
        axes[0].text(score_val + (abs(score_val)*0.01), bar.get_y() + bar.get_height()/2, 
                     f'Score: {score_val:.2f}', va='center', ha='left', fontsize=10, fontweight='bold')
    

    # Kanan: Bearish
    bars_bear = axes[1].barh(top_15_bear['Ticker'][::-1], top_15_bear['AI_Score'][::-1], color='#e74c3c', edgecolor='black', linewidth=0.5) 
    axes[1].set_title(f'🩸 Top {len(top_15_bear)} Weakest / Bearish Momentum', fontsize=16, fontweight='bold', pad=15)
    axes[1].grid(axis='x', linestyle='--', alpha=0.5)
    
    for bar in bars_bear:
        score_val = bar.get_width()
        if score_val < 0:
            axes[1].text(score_val - (abs(score_val)*0.01), bar.get_y() + bar.get_height()/2, 
                         f'Score: {score_val:.2f}', va='center', ha='right', fontsize=10, fontweight='bold')
        else:
            axes[1].text(score_val + (abs(score_val)*0.01), bar.get_y() + bar.get_height()/2, 
                         f'Score: {score_val:.2f}', va='center', ha='left', fontsize=10, fontweight='bold')

    plt.tight_layout()
    png_filename = f'{OUTPUT_DIR}/visualisasi_{MARKET_NAME}.png'
    plt.savefig(png_filename, dpi=150)
    print(f"[SUCCESS] Gambar Visualisasi disimpan di: {png_filename}")
    plt.show()

if __name__ == "__main__":
    run_inference_v2()