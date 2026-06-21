### FASE 1: Pabrik Data (Data Ingestion & Preprocessing)

**File Utama:** `build_custom_dataset_all.py`

Model secanggih apapun akan menjadi sampah jika diberi data sampah ( *Garbage In, Garbage Out* ). Di fase ini, kita memastikan AI mendapatkan makanan yang bergizi.

* **Proses:** *Script* ini membaca daftar 958 saham dari `ihsg_all.csv`, lalu mengunduh riwayat harganya dari Yahoo Finance.
* **Modifikasi Kunci (Kalender Dinamis):** Ingat masalah matriks yang menyusut menjadi 336 hari? Kita menyelesaikannya dengan membuat logika yang mencari saham tertua/terpanjang riwayatnya (misalnya `AKKU.JK` atau `GGRM.JK`) untuk dijadikan  **jangkar kalender** . Saham yang baru IPO (seperti `GOTO`) tidak akan merusak kalender ini, melainkan diisi dengan nilai kosong ( *Masking* ) pada hari-hari sebelum ia  *listing* .
* **Suntikan Fitur (CTA Touch):** Selain harga dasar (OHLCV), kita menyuntikkan indikator teknikal **RSI** sebagai fitur ke-6. Ini memberikan model "insting" tentang area jenuh beli ( *overbought* ) dan jenuh jual ( *oversold* ) yang biasa kamu gunakan dalam analisis manual.
* **Output:** Menghasilkan file matriks 3D `eod_data.pkl` berdimensi `(958 Saham, 1533 Hari, 6 Fitur)`.

---

### FASE 2: Otak AI (StockMixer Architecture & Training)

**File Utama:** `model.py` dan `train.py`

**Ini adalah jantung dari ***paper* akademik yang kamu bawa^^^^^^^^. StockMixer mendobrak tradisi lama. **Alih-alih menggunakan arsitektur rumit seperti RNN (LSTM) atau Graph Neural Networks (GNN) yang rawan ***overfitting* dan lambat ^^, ia menggunakan **MLP-Mixer** (Multi-Layer Perceptron murni) yang sangat ringan namun bertenaga^^^^^^^^.

**Cara berfikir otak ini dibagi menjadi tiga lapisan (** *Mixer Blocks* **)**^^^^^^^^:

1. **Indicator Mixing (Analisis Fitur):** Model mencampur 6 fitur (OHLCV + RSI) dalam satu hari untuk mencari pola tersembunyi^^^^^^^^. Misalnya, ia belajar bahwa  *"jika Shadow atas panjang dan RSI > 70, ini adalah anomali"* .
2. **Time Mixing (Analisis Tren 16 Hari):**    Model melihat ke belakang sejauh parameter `LOOKBACK = 16` hari. **Modifikasi jenius dari pembuat ***paper* ini adalah menggunakan **Upper Triangular Mask**^^^^^^^^. Artinya, model dilarang "mengintip" data masa depan saat mempelajari data masa lalu. **Ia juga memecah 16 hari ini menjadi beberapa skala/fragmen (misal 1 hari, 2 hari, 4 hari) untuk menangkap tren jangka pendek dan menengah sekaligus**^^^^^^^^^^^^^^^^^^.
3. **Stock Mixing (Analisis Rotasi Bandar):**    Ini yang membuat modelmu jenius. **Jika ada 958 saham, model tidak mencari hubungan satu-per-satu yang menghabiskan memori**^^. **Ia menggunakan konsep ***Stock-to-Market* dan *Market-to-Stock*^^^^^^^^^^^^^^^^^^.
   * *Logikanya:* Model melebur data 958 saham menjadi sebuah "Kondisi Pasar" (direpresentasikan oleh parameter `<span class="citation-10">market=20</span>` yang kita manipulasi kemarin agar tidak *error*^^^^^^^^). Lalu dari kondisi makro 20 dimensi ini, pengaruhnya dipantulkan kembali ke masing-masing 958 saham untuk melihat siapa yang paling diuntungkan dari kondisi tersebut.

* **Modifikasi Kunci (Early Stopping):** Di `train.py`, kita menambahkan "rem darurat". Model hanya akan menyimpan otak terbaiknya (`IDX_ALL_best.pth`) ke dalam folder `models/`. Jika selama 5 *epoch* berturut-turut performanya memburuk, ia akan berhenti berlatih untuk mencegah ia "menghafal" ( *overfitting* ).

---

### FASE 3: Eksekusi & Filter Realita (Inference & MLOps)

**File Utama:** `inference.py` dan `visualize_csv.py`

Fase ini adalah tempat di mana teoretis akademik diubah menjadi mesin pencetak uang ( *Alpha Generator* ) di dunia nyata.

* **Proses:** `inference.py` membangunkan otak dari file `.pth`, mengambil data 16 hari bursa terakhir, dan meminta AI memberikan **Conviction Score** (skor keyakinan momentum) untuk 958 saham.
* **Modifikasi Kunci (Risk Management Filter):** AI kadang terlalu polos. Ia menganggap saham tidur harga Rp50 yang naik 1 perak sebagai momentum 2% yang hebat. Kita menyuntikkan *Business Logic* di sini:
  1. *Script* mengambil Top 300 Saham Bullish dan Top 300 Saham Bearish.
  2. Mengecek data aslinya secara *live* ke Yahoo Finance.
  3. **Menendang** saham yang harganya gocap (Rp50) atau nilai transaksinya di bawah Rp 1 Miliar/hari.
* **Output Akhir:** 1. Sebuah file `rekomendasi_harian_lengkap.csv` yang berisi ratusan saham layak *trading* lengkap dengan peringkat AI-nya.
  2. *Script* `visualize_csv.py` yang membaca CSV tersebut dan menggambar **Horizontal Bar Chart** (Top 20 Bullish vs Top 20 Bearish). Visualisasi ini menghemat waktumu dari melakukan *screening* ratusan *chart* secara manual.

### Langkah Backtest Manual

Karena kamu akan melakukan *backtest* manual, rutinitas harianmu akan sangat sederhana:

1. Setelah bursa tutup (atau malam hari), *run* `build_custom_dataset_all.py` untuk meng- *update* data hari itu.
2. *Run* `inference.py` untuk mendapatkan prediksi besok.
3. *Run* `visualize_csv.py` untuk melihat gambarnya.
4. Keesokan harinya, kamu catat performa saham-saham di Top 10 Bullish tersebut.
