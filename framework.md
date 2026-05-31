### 1. Kotak Bawah Kanan: "Indicator & Time Mixing" (Fase Belajar Mandiri)

Sebelum murid-murid masuk ke kelas besar, mereka harus belajar sendirian dulu di rumah. Data mentah mereka adalah **$x$** (Buku pelajaran = Fitur/Indikator seperti harga Open, Close, RSI) dan **$T$** (Waktu = Belajar selama 16 hari terakhir).

* **Patch (Membagi Jam Belajar):**
  Kalau kamu disuruh menghafal buku tebal dalam semalam, pasti pusing. AI juga begitu. **Jadi, waktu belajarnya dipecah-pecah (** *Patching* **)**^^^^. Ada yang belajar pola harian (**$T$**), ada yang belajar pola per 2 hari (**$T/2$**), dan seterusnya. **Ini supaya AI bisa melihat tren jangka pendek dan menengah**^^^^^^^^.
* **Indicator Mixing (Menggabungkan Pelajaran):**
  **Di fase ini, si murid mencoba mencari hubungan antar mata pelajaran di hari yang sama**^^. **Misalnya: ***"Oh, kalau pelajaran RSI-nya tinggi, biasanya pelajaran Harga-nya juga lagi di atas"*^^.
* **Time Mixing (Mengingat Memori):**
  Ini bagian paling pintar. **Murid mencoba mengingat runtutan kejadian dari hari 1 sampai hari 16. Tapi ada aturan ketat: ****Murid tidak boleh melihat kunci jawaban masa depan**^^. Ingatan hari ke-5 hanya boleh dipengaruhi oleh hari ke-1 sampai 4. (Di kode yang kita bahas sebelumnya, ini disebut  *Upper Triangular Mask* ).
* **Hasil Akhir Fase Ini (**$h$**):**
  **Semua hasil belajar mandiri itu digabung menjadi sebuah "Buku Catatan Pribadi" yang sangat tebal (simbol **$h$**) untuk masing-masing murid**^^.

### 2. Kotak Atas Kanan: "Stock Mixing" (Fase Diskusi Kelas)

Sekarang, 958 murid yang sudah bawa "Buku Catatan Pribadi" (**$h$**) ini masuk ke satu kelas besar (Pasar Saham).

Murid-murid ini tidak hidup sendirian. **Nilai mereka pasti terpengaruh oleh suasana kelas**^^^^^^^^. **Misalnya, kalau kelasnya lagi semangat (Bull Market), murid yang bodoh pun nilainya ikut naik**^^. Kalau kelasnya lagi malas (Bear Market), murid pintar pun bisa ikut anjlok.

Lalu bagaimana cara mereka berdiskusi?

* **Cara Bodoh (Bikin Overfitting):** Membiarkan 958 murid saling ngobrol satu lawan satu. Ada ratusan ribu obrolan! **Kelas jadi sangat berisik, AI jadi pusing, dan akhirnya cuma "menghafal" tanpa paham (** *Overfitting* **)**^^.
* **Cara Pintar StockMixer (Bagian tengah kotak):** Alih-alih saling ngobrol, ke-958 murid ini diminta mengumpulkan ringkasan catatannya ke depan kelas, lalu disatukan menjadi **"Suasana Kelas"** (diwakili oleh kotak kecil bernama **$m$**, di mana kita menggunakan angka 20)^^. **Proses ini disebut ***Stock-to-Market*^^^^^^^^.
  Setelah itu, guru mengambil "Suasana Kelas" yang berisi 20 poin utama ini, dan membagikannya kembali ke 958 murid. **Proses ini disebut ***Market-to-Stock*^^^^^^^^.

**Skip Connection (Garis Panah Melengkung di Atas):**
AI sadar bahwa diskusi kelas itu penting, tapi kepribadian asli si murid jangan sampai hilang. **Jadi, hasil diskusi kelas tadi ditambahkan kembali dengan "Buku Catatan Pribadi" asli si murid**^^.

### 3. Bagian Kiri: Gambaran Besar & Tebakan Akhir (FC)

Sekarang kamu lihat gambar paling kiri secara utuh:

1. Murid **$X_1, X_2,$** sampai **$X_N$** belajar mandiri di rumah (Indicator & Time Mixing).
2. Mereka membawa catatannya ke sekolah (**$h_1, h_2,$** sampai **$h_N$**).
3. Mereka berdiskusi dengan cerdas di kelas tanpa saling berisik (Stock Mixing).
4. **Tahap Akhir (FC - Fully Connected):** Guru (Model AI) melihat hasil akhir dari setiap murid, lalu memberikan tebakan nilai akhir^^. *"Murid **$X_1$** saya prediksi nilainya +15,000, murid **$X_2$** nilainya -5,000!"*

**Jadi, algoritma StockMixer ini intinya adalah memisahkan cara AI belajar menjadi 3 langkah yang sangat rapi: ****Pahami dulu data per harinya, gabungkan ingatannya secara waktu, lalu biarkan mereka saling mempengaruhi secara pasar.** Pendekatan yang sederhana inilah yang membuatnya jauh lebih tangguh dan tidak mudah *error* dibandingkan arsitektur *Deep Learning* lain yang terlalu rumit^^^^^^^^.
