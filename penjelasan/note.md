1. Olah Data harian saat market close
2. train data ulang setiap hari
3. cek predictive setiap hari
4. filtering saham tidur


A. Cara Run, biasakan run di sore hari, dan eksekusi di market pagi hari


* update tanggal build_custom_dataset.py
* run pytonh build_custom_dataset.py
* run train.py
* run inference.py
* venv\Scripts\activate   (kalau belum aktiv venv nya)


### B. Cara Mengubahnya Menjadi Keputusan Eksekusi

Hasil keluaran dari StockMixer ini adalah bahan mentah yang sempurna untuk disuntikkan ke dalam aplikasi *screener* saham milikmu. Alur eksekusi yang ideal adalah:

1. **Terima Top 10:** Ambil 10 nama dari grafik ini setiap bursa tutup.
2. **Saring Likuiditas:** Buang saham yang nilai transaksi hariannya ( *Value* ) di bawah standar toleransimu (misalnya buang MKNT, ITMA, atau INTD jika tidak masuk kriteria likuid).
3. **Konfirmasi Teknikal:** Untuk saham berkapitalisasi besar yang lolos (seperti BBCA.JK atau GGRM.JK), periksa apakah posisinya sedang berada di area *support* atau  *breakout* .
4. **Alokasi Dana:** Beli saham-saham yang lolos filter di keesokan paginya.
