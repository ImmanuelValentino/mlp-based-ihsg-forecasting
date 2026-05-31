import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')

# --- KONFIGURASI AMAN ---
CSV_FILE = '../rekomendasi_harian_lengkap.csv'
TOP_N = 20 # Jumlah saham yang ingin ditampilkan (bisa kamu ubah jadi 30 atau 50)

def visualize_horizontal():
    print(f"Membaca data dari: {CSV_FILE}...")
    if not os.path.exists(CSV_FILE):
        print(f"[ERROR] File {CSV_FILE} tidak ditemukan. Jalankan inference.py terlebih dahulu.")
        return

    # 1. Baca Data
    df = pd.read_csv(CSV_FILE)
    
    # 2. Keamanan: Pastikan data diurutkan dari skor AI tertinggi ke terendah
    df = df.sort_values(by='AI_Score', ascending=False).reset_index(drop=True)

    if len(df) < TOP_N * 2:
        print(f"[WARNING] Jumlah saham valid di CSV ({len(df)}) kurang dari {TOP_N*2}. Grafik mungkin terlihat kosong.")

    # 3. Pisahkan kutub atas (Bullish) dan kutub bawah (Bearish)
    df_top = df.head(TOP_N)
    df_bottom = df.tail(TOP_N)

    # 4. Bangun Frame Kanvas (1 baris, 2 kolom)
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))

    # --- SUBPLOT 1: STRONG BULLISH ---
    # Catatan: Barh menggambar dari bawah ke atas. Kita reverse index [::-1] agar Rank 1 ada di paling atas.
    bars_bull = axes[0].barh(df_top['Ticker'][::-1], df_top['AI_Score'][::-1], color='#2ecc71', edgecolor='black', linewidth=0.5)
    axes[0].set_title(f'🚀 Top {TOP_N} Strong Bullish Momentum', fontsize=16, fontweight='bold', pad=15)
    axes[0].set_xlabel('AI Conviction Score (Sumbu X)', fontsize=12)
    axes[0].set_ylabel('Kode Saham', fontsize=12)
    axes[0].grid(axis='x', linestyle='--', alpha=0.5)
    
    # Tambahkan angka di ujung bar
    for bar in bars_bull:
        width = bar.get_width()
        axes[0].text(width + (width*0.01), bar.get_y() + bar.get_height()/2, f'{int(width)}', 
                     va='center', ha='left', fontsize=9)

    # --- SUBPLOT 2: STRONG BEARISH ---
    # Balik juga agar Rank terburuk ada di paling bawah
    bars_bear = axes[1].barh(df_bottom['Ticker'][::-1], df_bottom['AI_Score'][::-1], color='#e74c3c', edgecolor='black', linewidth=0.5)
    axes[1].set_title(f'🩸 Top {TOP_N} Strong Bearish (Avoid / Sell)', fontsize=16, fontweight='bold', pad=15)
    axes[1].set_xlabel('AI Conviction Score (Sumbu X)', fontsize=12)
    axes[1].grid(axis='x', linestyle='--', alpha=0.5)

    # Tambahkan angka di ujung bar merah (karena negatif, angkanya ada di kiri)
    for bar in bars_bear:
        width = bar.get_width()
        axes[1].text(width - (abs(width)*0.01), bar.get_y() + bar.get_height()/2, f'{int(width)}', 
                     va='center', ha='right', fontsize=9)

    # --- FINALISASI TAMPILAN ---
    plt.tight_layout()
    output_path = '../horizontal_recommendation.png'
    plt.savefig(output_path, dpi=150)
    print(f"[SUCCESS] Visualisasi horizontal berhasil di-render dan disimpan di: {output_path}")
    plt.show()

if __name__ == '__main__':
    visualize_horizontal()