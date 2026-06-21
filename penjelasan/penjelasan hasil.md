### 1. "10 Prediction" (Grafik Hijau Pertama - Raw AI)

Ini adalah murni isi "otak" si AI StockMixer sebelum kita beri kacamata realita.

* **Cara kerjanya:** AI hanya melihat deretan angka di file `.pkl` kamu. Jika ia melihat sebuah saham yang harganya Rp 50 tiba-tiba naik ke Rp 51, AI menghitung itu sebagai lonjakan momentum 2% dalam sehari. AI sangat mengagumi angka persentase ekstrem seperti itu.
* **Hasilnya:** Muncul nama-nama anomali seperti **MKNT.JK** (saham gocap/tidur),  **UNTD.JK** ,  **INTD.JK** , dan  **IBST.JK** .
* **Masalah di dunia nyata:** Jika kamu mencoba mengeksekusi rekomendasi ini, kamu mungkin tidak bisa membelinya karena antrean jualnya ( *offer* ) kosong, sahamnya disuspensi bursa, atau masuk papan FCA (lelang berkala). Prediksinya benar secara rumus, tapi **tidak bisa dieksekusi** (untreadable).

### 2. "10 Filtered" (Grafik Biru & Hijau/Merah - Risk Managed)

Ini adalah hasil setelah kita menyuntikkan *Business Logic* atau "Logika Bandar" ke dalam kode `inference.py`.

* **Cara kerjanya:** AI tetap memberikan peringkat, tetapi sebelum ditampilkan ke layar, sistem menyeleksinya ke bursa (via  *yfinance* ). Sistem kita bertanya: *"Apakah saham ini harganya di atas Rp 50? Apakah ada transaksi minimal Rp 1 Miliar hari ini?"*
* **Hasilnya:** Saham-saham "halu" seperti MKNT dan UNTD langsung ditendang dari daftar. Kekosongan kursi Top 10 itu langsung diisi oleh saham-saham raksasa beneran yang punya likuiditas masif: **BBCA.JK (BCA), AALI.JK (Astra Agro Lestari), INCO.JK (Vale), ULTJ.JK (Ultra Milk), dan ASII.JK (Astra).**
* **Keunggulan di dunia nyata:** Kesepuluh saham yang muncul di grafik *Filtered* ini  **100% bisa langsung kamu beli besok pagi** . Uang Rp 100 Juta pun bisa masuk dan keluar dengan mudah tanpa takut nyangkut karena  *bid/offer* -nya tebal.

**Kesimpulan (Analogi):**

* **10 Prediction** ibarat seorang profesor matematika jenius yang jago rumus tapi belum pernah turun langsung ke lantai bursa.
* **10 Filtered** ibarat profesor matematika tersebut didampingi oleh seorang *Broker* senior yang menyeleksi mana tebakan profesor yang masuk akal untuk dibeli pakai uang sungguhan.
