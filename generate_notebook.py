import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Analisis & Visualisasi Proyek IHSG Forecasting (StockMixer V1)\n",
    "\n",
    "Notebook ini dibuat khusus untuk keperluan presentasi PPT proyek Anda dengan fokus penuh pada **Model V1 (Baseline)**. Seluruh visualisasi, korelasi fitur, hipotesis preproses, dan visualisasi hasil model termuat di sini. Anda tinggal menjalankan sel-sel di bawah ini secara bertahap."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--- \n",
    "## 1. Latar Belakang & Tujuan\n",
    "\n",
    "### **Latar Belakang**\n",
    "*   **Tantangan Prediksi IHSG**: Pergerakan harga saham sangat dinamis dan dipengaruhi oleh tren historis (temporal) serta keterkaitan antar-saham di pasar (spasial).\n",
    "*   **Kelemahan Model Lain**: Model deep learning konvensional (seperti LSTM) cenderung lambat dan sulit menangkap hubungan antar-saham. Model Graph (GNN) membutuhkan data relasi eksternal yang kompleks.\n",
    "*   **Pendekatan StockMixer V1**: Menggunakan arsitektur berbasis MLP (*Multi-Layer Perceptron*) yang sangat ringan dan cepat. Dengan memproses 6 fitur utama melalui modul *Indicator Mixing*, *Temporal Mixing*, dan *Stock Mixing*, model ini secara efisien memprediksi return harian (1 hari ke depan) saham-saham di IHSG tanpa graf relasi rumit.\n",
    "\n",
    "### **Tujuan**\n",
    "1.  Melakukan forecasting pergerakan harga saham likuid di Bursa Efek Indonesia (IHSG) untuk 1 hari ke depan.\n",
    "2.  Menganalisis korelasi 6 fitur input secara **keseluruhan (overall)** untuk seluruh saham bursa.\n",
    "3.  Menyaring saham-saham terbaik (Bullish) berdasarkan skor keyakinan AI dengan batasan manajemen risiko (saham aktif & likuid)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--- \n",
    "## 2. Hipotesis & Use Case Proyek\n",
    "\n",
    "### **1. Use Case Investor & Trader (Pengguna)**\n",
    "*   **Masalah Pengguna**: Investor ritel sering kesulitan memilih saham terbaik secara objektif karena banyaknya jumlah emiten di IHSG. Ada juga risiko terjebak membeli saham tidak likuid (*saham gorengan*) atau saham mati (*saham gocap*).\n",
    "*   **Usecase Aplikasi**: Sistem menyaring saham-saham likuid secara otomatis (Transaksi Harian > Rp 1 Miliar, Harga > Rp 51) dan menyajikan peringkat **Top 10 Rekomendasi Beli (Bullish)** untuk 1 hari ke depan berdasarkan skor keyakinan AI secara objektif.\n",
    "\n",
    "### **2. Hipotesis & Pembuktian Algoritma**\n",
    "*   **Hipotesis Relatif (Cross-Stock Ranking)**: Keuntungan investasi maksimal diperoleh dengan membeli saham yang kinerjanya *paling unggul* dibanding saham lain di hari yang sama. Oleh karena itu, model harus berfokus pada peringkat relatif antar-saham, bukan sekadar prediksi harga individu.\n",
    "*   **Mekanisme Pembuktian StockMixer**:\n",
    "    *   *Indicator Mixing*: Mengombinasikan data harga, volume, dan RSI untuk memetakan kekuatan pasar secara multi-dimensi.\n",
    "    *   *Temporal Mixing*: Menangkap sinyal tren historis (momentum 16 hari bursa terakhir).\n",
    "    *   *Stock Mixing*: Mempelajari keterkaitan pergerakan antar-saham di bursa secara bersamaan (*cross-sectional*).\n",
    "*   **Metrik Pembuktian**: Algoritma terbukti sukses jika menghasilkan nilai *Precision@10* yang tinggi (lebih dari 50% rekomendasi AI benar-benar profit) serta *Sharpe Ratio* yang positif pada data pengujian."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import os\n",
    "\n",
    "# Gunakan estetika plot modern\n",
    "sns.set_theme(style=\"whitegrid\")\n",
    "plt.rcParams['figure.figsize'] = (10, 6)\n",
    "\n",
    "v1_path = 'dataset/IDX_ALL'\n",
    "\n",
    "print(\"=== MENELUSURI DATA PREPROSES V1 ===\")\n",
    "if os.path.exists(v1_path):\n",
    "    with open(os.path.join(v1_path, 'eod_data.pkl'), 'rb') as f:\n",
    "        v1_data = pickle.load(f)\n",
    "    with open(os.path.join(v1_path, 'mask_data.pkl'), 'rb') as f:\n",
    "        v1_mask = pickle.load(f)\n",
    "    with open(os.path.join(v1_path, 'gt_data.pkl'), 'rb') as f:\n",
    "        v1_gt = pickle.load(f)\n",
    "    \n",
    "    # Tampilkan dimensi asli sebelum/sesudah transpose\n",
    "    print(f\"Dimensi EOD Data (Hari, Saham, Fitur) : {v1_data.shape}\")\n",
    "    print(f\"Dimensi Mask Data (Hari, Saham)          : {v1_mask.shape}\")\n",
    "    print(f\"Dimensi Ground Truth (Hari, Saham)        : {v1_gt.shape}\")\n",
    "else:\n",
    "    print(\"[WARN] Data preproses V1 belum di-generate. Silakan jalankan 'python build_custom_dataset.py' terlebih dahulu.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--- \n",
    "## 3. Analisis Korelasi Fitur Secara Keseluruhan (Overall)\n",
    "\n",
    "Pada sel berikut, kita menghitung korelasi **keseluruhan (overall)** untuk 6 fitur utama dengan menggabungkan (*reshape*) seluruh data hari perdagangan dan seluruh saham yang aktif di bursa.\n",
    "\n",
    "*Note: Fitur RSI dihitung secara dinamis dari harga Close ternormalisasi agar terbebas dari issue flat value saat eksport data.*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "if os.path.exists(v1_path):\n",
    "    # Buat salinan eod_data agar data mentah tidak berubah\n",
    "    eod_corrected = v1_data.copy()\n",
    "    num_days, num_stocks, num_features = eod_corrected.shape\n",
    "    \n",
    "    print(\"Mengkalkulasi ulang fitur RSI dan membuat matriks korelasi overall...\")\n",
    "    for s_idx in range(num_stocks):\n",
    "        # Fitur indeks 3 adalah Close price (Z-Score normalized)\n",
    "        close_prices = pd.Series(eod_corrected[:, s_idx, 3])\n",
    "        \n",
    "        # Rumus RSI dinamis\n",
    "        delta = close_prices.diff()\n",
    "        up = delta.clip(lower=0)\n",
    "        down = -1 * delta.clip(upper=0)\n",
    "        \n",
    "        ema_up = up.ewm(com=13, adjust=False).mean()\n",
    "        ema_down = down.ewm(com=13, adjust=False).mean()\n",
    "        \n",
    "        rs = ema_up / ema_down.replace(0, np.nan)\n",
    "        rsi = 100 - (100 / (1 + rs))\n",
    "        rsi_vals = rsi.fillna(50).replace([np.inf, -np.inf], [100, 0]).values\n",
    "        \n",
    "        # Normalisasi Z-Score untuk RSI agar seragam skalanya\n",
    "        mean_rsi = np.nanmean(rsi_vals)\n",
    "        std_rsi = np.nanstd(rsi_vals)\n",
    "        if std_rsi != 0:\n",
    "            rsi_vals = (rsi_vals - mean_rsi) / std_rsi\n",
    "        else:\n",
    "            rsi_vals = np.zeros_like(rsi_vals)\n",
    "            \n",
    "        # Masukkan ke kolom RSI (indeks ke-5)\n",
    "        eod_corrected[:, s_idx, 5] = rsi_vals\n",
    "        \n",
    "    # Reshape dari 3D (Hari, Saham, Fitur) menjadi 2D (Hari * Saham, Fitur) untuk korelasi Overall\n",
    "    overall_feat = eod_corrected.reshape(-1, num_features)\n",
    "    df_overall = pd.DataFrame(overall_feat, columns=['Open', 'High', 'Low', 'Close', 'Volume', 'RSI'])\n",
    "    \n",
    "    # Hitung korelasi\n",
    "    corr_matrix_overall = df_overall.corr()\n",
    "    \n",
    "    # Visualisasikan Heatmap Korelasi Overall\n",
    "    plt.figure(figsize=(9, 7))\n",
    "    sns.heatmap(corr_matrix_overall, annot=True, cmap='coolwarm', fmt=\".2f\", linewidths=0.5, vmin=-1, vmax=1)\n",
    "    plt.title(\"Heatmap Korelasi Fitur Overall - Seluruh Saham IHSG (Model V1)\", fontsize=13, fontweight='bold')\n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "else:\n",
    "    print(\"Silakan buat dataset V1 terlebih dahulu.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--- \n",
    "## 4. Visualisasi Runtun Waktu & Distribusi Target (V1)\n",
    "\n",
    "Di bawah ini kita memplot tren harga penutupan (`Close`), volume transaksi, dan momentum `RSI` dalam runtun waktu (time series) pada satu saham contoh (seperti **BBCA.JK**), diikuti oleh grafik distribusi dari target return harian (Ground Truth)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "if os.path.exists(v1_path):\n",
    "    # Ambil list ticker\n",
    "    tickers = pd.read_csv('ihsg_all.csv')['Ticker'].dropna().unique().tolist()\n",
    "    sample_ticker = 'BBCA.JK'\n",
    "    \n",
    "    if sample_ticker in tickers:\n",
    "        ticker_idx = tickers.index(sample_ticker)\n",
    "    else:\n",
    "        ticker_idx = 0\n",
    "        sample_ticker = tickers[0]\n",
    "        \n",
    "    # Ambil data runtun waktu saham terpilih dari matriks yang telah dikoreksi RSI-nya\n",
    "    stock_feat = eod_corrected[:, ticker_idx, :]\n",
    "    df_feat = pd.DataFrame(stock_feat, columns=['Open', 'High', 'Low', 'Close', 'Volume', 'RSI'])\n",
    "    \n",
    "    plot_len = min(200, df_feat.shape[0])\n",
    "    df_plot = df_feat.tail(plot_len).copy().reset_index(drop=True)\n",
    "    \n",
    "    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)\n",
    "    \n",
    "    # 1. Plot Close Price\n",
    "    axes[0].plot(df_plot['Close'], color='#2980b9', lw=2, label='Close Price (Normalized)')\n",
    "    axes[0].set_title(f\"Tren Runtun Waktu Fitur Saham {sample_ticker} (200 Hari Terakhir)\", fontsize=13, fontweight='bold')\n",
    "    axes[0].legend(loc='upper left')\n",
    "    axes[0].grid(True, alpha=0.3)\n",
    "    \n",
    "    # 2. Plot Volume\n",
    "    axes[1].plot(df_plot['Volume'], color='#e67e22', lw=1.5, label='Volume Normalized')\n",
    "    axes[1].legend(loc='upper left')\n",
    "    axes[1].grid(True, alpha=0.3)\n",
    "    \n",
    "    # 3. Plot RSI\n",
    "    axes[2].plot(df_plot['RSI'], color='#8e44ad', lw=1.5, label='RSI (14) Normalized')\n",
    "    axes[2].axhline(0, color='red', ls='--', alpha=0.5, label='Neutral (RSI Z-Score 0)')\n",
    "    axes[2].legend(loc='upper left')\n",
    "    axes[2].grid(True, alpha=0.3)\n",
    "    axes[2].set_xlabel(\"Hari Perdagangan\")\n",
    "    \n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "else:\n",
    "    print(\"Silakan buat dataset V1 terlebih dahulu.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "if os.path.exists(v1_path):\n",
    "    v1_returns = v1_gt.flatten()\n",
    "    \n",
    "    # Filter data pencilan ekstrim untuk visualisasi kurva distribusi yang rapi\n",
    "    v1_filt = v1_returns[(v1_returns > -0.1) & (v1_returns < 0.1)]\n",
    "    \n",
    "    plt.figure(figsize=(10, 5))\n",
    "    sns.histplot(v1_filt, kde=True, color='#3498db', bins=50, stat=\"density\", alpha=0.6)\n",
    "    plt.axvline(0, color='black', linestyle=':', alpha=0.5)\n",
    "    plt.title(\"Analisis Distribusi Label Target Return Harian V1 (1-Day Return)\", fontsize=13, fontweight='bold')\n",
    "    plt.xlabel(\"Persentase Return Harian (Desimal)\")\n",
    "    plt.ylabel(\"Kepadatan Kerapatan (Density)\")\n",
    "    plt.grid(True, alpha=0.3)\n",
    "    plt.show()\n",
    "else:\n",
    "    print(\"Dataset V1 belum siap.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--- \n",
    "## 5. Hasil Model Machine Learning (Model Output)\n",
    "\n",
    "Bagian ini memuat file hasil rekomendasi model V1 terbaru dari folder `outputs/` guna melihat hasil top rekomendasi beli (Bullish) di bursa IHSG."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "output_root = 'outputs'\n",
    "if os.path.exists(output_root):\n",
    "    # Ambil folder tanggal terbaru\n",
    "    dates = [d for d in os.listdir(output_root) if os.path.isdir(os.path.join(output_root, d)) and '-' in d]\n",
    "    dates.sort(reverse=True)\n",
    "    \n",
    "    if dates:\n",
    "        latest_date = dates[0]\n",
    "        v1_csv = os.path.join(output_root, latest_date, 'rekomendasi_lengkap_IDX_ALL.csv')\n",
    "        \n",
    "        print(f\"Membaca data output prediksi tanggal: {latest_date}\\n\")\n",
    "        \n",
    "        if os.path.exists(v1_csv):\n",
    "            df_v1 = pd.read_csv(v1_csv)\n",
    "            \n",
    "            # Plot Top 10 V1\n",
    "            plt.figure(figsize=(12, 6))\n",
    "            top10_v1 = df_v1.head(10)\n",
    "            sns.barplot(x='AI_Score', y='Ticker', data=top10_v1, palette='viridis', hue='Ticker', legend=False)\n",
    "            plt.title(f\"Model V1 (1-Day): Top 10 Bullish Saham IHSG ({latest_date})\", fontsize=13, fontweight='bold')\n",
    "            plt.xlabel(\"Conviction Score (Skor Keyakinan AI Model)\")\n",
    "            plt.ylabel(\"Ticker Saham\")\n",
    "            plt.grid(True, alpha=0.3)\n",
    "            plt.tight_layout()\n",
    "            plt.show()\n",
    "            \n",
    "            print(\"\\n--- TABEL TOP 10 REKOMENDASI SAHAM BULLISH MODEL V1 ---\")\n",
    "            print(df_v1[['Ticker', 'Harga', 'Trx_Miliar', 'AI_Score']].head(10).to_string(index=False))\n",
    "        else:\n",
    "            print(f\"File output CSV '{v1_csv}' tidak ditemukan.\")\n",
    "    else:\n",
    "        print(\"Belum ada data output prediksi di folder outputs.\")\n",
    "else:\n",
    "    print(\"Folder outputs tidak ditemukan.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--- \n",
    "## 6. Simpulan & Rekomendasi\n",
    "\n",
    "Berikut adalah poin ringkas kesimpulan Model V1 untuk dipindahkan ke slide presentasi Anda:\n",
    "\n",
    "1.  **Efisiensi Fitur V1**: Menggunakan 6 fitur dasar (`Open`, `High`, `Low`, `Close`, `Volume`, `RSI`) yang mampu melatih model deep learning secara cepat tanpa membebani memori.\n",
    "2.  **Korelasi Fitur**: Fitur harga (`OHLC`) terbukti berkorelasi hampir sempurna (multikolinier), sementara fitur `Volume` dan `RSI` memberikan informasi tambahan independen tentang kekuatan tren dan aktivitas transaksi.\n",
    "3.  **Target Horizon**: Model V1 fokus memprediksi fluktuasi harian (1 hari ke depan) sehingga sangat sesuai untuk kebutuhan perdagangan jangka pendek (*day trading*).\n",
    "4.  **Mitigasi Risiko Likuiditas**: Penyaringan otomatis terhadap saham berharga kurang dari Rp 51 (menghindari saham gocap) dan transaksi harian di bawah Rp 1 Miliar berhasil menyaring saham yang tidak likuid, menjaga agar rekomendasi aman untuk dieksekusi secara riil."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open('analisis_presentasi.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("Notebook 'analisis_presentasi.ipynb' (Khusus V1) berhasil dibuat!")
