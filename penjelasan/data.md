ALUR BIG DATA



### Tahap 1: Extraction (Pengambilan Data Mentah)

Di sini kita menarik ribuan hari data untuk 958 saham secara massal.

**Implementasi di Kode:**
Kita menggunakan *library* `yfinance` untuk mengunduh data secara paralel (dikelompokkan per  *ticker* ).

**Python**

```
# Membaca daftar ticker dari CSV
df_tickers = pd.read_csv('ihsg_all.csv')
valid_tickers = df_tickers['Ticker'].dropna().unique().tolist()

print(f"[INFO] Mengunduh data untuk {len(valid_tickers)} saham...")

# Penarikan data massal via API Yahoo Finance
all_raw_data = yf.download(
    valid_tickers, 
    start="2018-01-01", # Ditarik mundur bertahun-tahun
    end="2026-05-14", 
    group_by='ticker', 
    auto_adjust=True
)
```

---

### Tahap 2: Transformation & Preprocessing (Pemrosesan Data)

#### 2.1 Time-Series Alignment (Penyelarasan Kalender)

Ini adalah baris kode dinamis yang kita perbaiki kemarin agar matriksmu tidak menyusut menjadi 336 hari akibat saham IPO baru. Kita mencari "Jangkar Master".

**Implementasi di Kode:**

**Python**

```
# 1. Cari saham dengan baris data terbanyak (paling tua)
longest_ticker = max(valid_tickers, key=lambda t: len(all_raw_data[t].dropna(how='all')))

# 2. Jadikan tanggal-tanggal di saham tertua ini sebagai 'Master Calendar'
reference_df = all_raw_data[longest_ticker].dropna(how='all')
master_dates = reference_df.index
```

#### 2.2 Feature Engineering (Suntikan RSI)

Kita menambahkan indikator teknikal agar AI punya "insting"  *overbought/oversold* .

**Implementasi di Kode:**

**Python**

```
# Menghitung perbedaan harga dari hari ke hari
delta = df['Close'].diff()

# Memisahkan hari saat untung (gain) dan rugi (loss)
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

# Menghitung Relative Strength dan RSI
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# Mengisi hari-hari awal yang RSI-nya masih kosong (NaN) dengan angka netral 50
df['RSI'] = df['RSI'].fillna(50) 
```

#### 2.3 Ground Truth Generation (Kunci Jawaban untuk AI)

Kita menghitung persentase *return* (keuntungan/kerugian) untuk target prediksi.

**Implementasi di Kode:**

**Python**

```
# Rumus (Harga Hari Ini - Harga Kemarin) / Harga Kemarin
df['Return'] = df['Close'].pct_change()

# PENTING: Karena kita memprediksi besok, kita harus menggeser (shift) 
# data return besok ke baris hari ini sebagai 'Ground Truth'
gt_array = df['Return'].shift(-1).values
```

#### 2.4 Z-Score Normalization & Masking Data

Kita mengubah skala harga (menjadi rentang 0-1) dan membuat "topeng" pelindung untuk saham yang belum lahir (IPO).

**Implementasi di Kode:**

**Python**

```
# 1. Pembuatan Masking (1 = Ada data, 0 = Saham belum IPO / Suspend)
# Mengecek apakah di tanggal tersebut harga 'Close' ada isinya
mask_array = (~df['Close'].isna()).astype(float).values

# 2. Z-Score Normalization (Standarisasi Skala)
# Memilih 6 kolom fitur: Open, High, Low, Close, Volume, RSI
features = df[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI']]

# Mengurangi harga dengan rata-rata, lalu dibagi standar deviasi
normalized_features = (features - features.mean()) / features.std()

# Mengisi angka kosong akibat saham belum IPO dengan angka 0
normalized_features = normalized_features.fillna(0)
```

---

### Tahap 3: Load (Penyusunan Tensor Matriks 3D & Pickle)

Setelah semuanya dihitung dan diseragamkan per saham, kita menumpuk ratusan tabel tersebut menjadi satu blok Tensor raksasa dan menyimpannya.

**Implementasi di Kode:**

**Python**

```
# Menyatukan list data per saham menjadi Numpy Array (Tensor 3D)
# Bentuk (Shape) akhirnya: (Total Hari, Total Saham, 6 Fitur)
eod_matrix = np.array(all_eod_data) 
mask_matrix = np.array(all_mask_data)
gt_matrix = np.array(all_gt_data)

print(f"[SUCCESS] Bentuk Matriks EOD : {eod_matrix.shape}")

# Serialize dan simpan bongkahan data ke dalam format .pkl (Pickle)
# Format ini sangat cepat dibaca oleh PyTorch saat proses training
import pickle
import os

dataset_path = "dataset/IDX_ALL"
os.makedirs(dataset_path, exist_ok=True)

with open(os.path.join(dataset_path, "eod_data.pkl"), "wb") as f:
    pickle.dump(eod_matrix, f)

with open(os.path.join(dataset_path, "mask_data.pkl"), "wb") as f:
    pickle.dump(mask_matrix, f)

with open(os.path.join(dataset_path, "gt_data.pkl"), "wb") as f:
    pickle.dump(gt_matrix, f)
```
