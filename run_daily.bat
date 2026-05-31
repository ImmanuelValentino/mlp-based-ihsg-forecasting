@echo off
echo ===================================================
echo Memulai Pipeline StockMixer Harian (V2 & V3)
echo ===================================================

cd /d E:\Coding\BigData\StockMixer

echo 1. Mengaktifkan Virtual Environment...
call venv\Scripts\activate.bat

echo 2. Generate Data Terbaru V2...
python build_custom_dataset_v2.py

echo 3. Generate Data Terbaru V3...
python build_custom_dataset_v3.py

echo 4. Jalankan AI Inference V2 (Target 1 Hari)...
python src\inference_v2.py

echo 5. Jalankan AI Inference V3 (Target 3 Hari)...
python src\inference_v3.py
REM (Opsional: Kamu bisa membuat salinan inference khusus V3 jika ingin run bersamaan)
REM python src\inference_v3.py

echo ===================================================
echo Selesai! Hasil tersimpan di folder outputs.
echo ===================================================
pause