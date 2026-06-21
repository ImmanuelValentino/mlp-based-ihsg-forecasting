import os
import sys
import subprocess
import datetime
import csv
import json
import threading
import queue
from flask import Flask, render_template, jsonify, send_file, Response, request

app = Flask(__name__, static_folder='static', template_folder='templates')

# Global state for pipeline
pipeline_status = {
    'is_running': False,
    'current_step': '',
    'date': '',
    'success': False,
    'error_message': '',
    'model_version': 'v1'
}
log_queue = queue.Queue()
log_history = []

def run_command_in_background(cmd, step_name):
    global pipeline_status
    pipeline_status['current_step'] = step_name
    msg = f"\n>>> [STEP: {step_name}] Menjalankan: {' '.join(cmd)}\n"
    log_queue.put(msg)
    log_history.append(msg)
    
    try:
        # Menjalankan subprocess dengan menangkap stdout & stderr secara real-time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            shell=True
        )
        
        # Baca output baris demi baris
        for line in process.stdout:
            log_queue.put(line)
            log_history.append(line)
            
        process.wait()
        
        if process.returncode != 0:
            err_msg = f"ERROR: Langkah '{step_name}' gagal dengan kode keluar {process.returncode}\n"
            log_queue.put(err_msg)
            log_history.append(err_msg)
            return False
        return True
    except Exception as e:
        err_msg = f"EXCEPT: Gagal menjalankan '{step_name}': {str(e)}\n"
        log_queue.put(err_msg)
        log_history.append(err_msg)
        return False

def pipeline_thread_worker(target_date, model_version):
    global pipeline_status, log_history
    pipeline_status['is_running'] = True
    pipeline_status['date'] = target_date
    pipeline_status['success'] = False
    pipeline_status['error_message'] = ''
    pipeline_status['model_version'] = model_version
    
    # Reset log
    log_history.clear()
    while not log_queue.empty():
        try:
            log_queue.get_nowait()
        except queue.Empty:
            break
            
    start_msg = f"=== MEMULAI PIPELINE STOCKMIXER {model_version.upper()} UNTUK TANGGAL: {target_date} ===\n"
    log_queue.put(start_msg)
    log_history.append(start_msg)
    
    # Path Python venv
    base_dir = os.path.dirname(os.path.abspath(__file__))
    python_exe = os.path.join(base_dir, 'venv', 'Scripts', 'python.exe')
    if not os.path.exists(python_exe):
        python_exe = sys.executable  # Fallback ke python saat ini jika venv tidak terdeteksi
        
    # Daftar perintah pipeline sesuai versi model
    if model_version == 'v3':
        steps = [
            {
                'name': 'Generate Dataset V3',
                'cmd': [python_exe, 'build_custom_dataset_v3.py', '--end-date', target_date]
            },
            {
                'name': 'AI Inference V3',
                'cmd': [python_exe, os.path.join('src', 'inference_v3.py'), '--date', target_date]
            }
        ]
    else: # default to v1
        steps = [
            {
                'name': 'Generate Dataset V1',
                'cmd': [python_exe, 'build_custom_dataset.py', '--end-date', target_date]
            },
            {
                'name': 'AI Inference V1',
                'cmd': [python_exe, os.path.join('src', 'inference.py'), '--date', target_date]
            }
        ]
    
    success = True
    for step in steps:
        if not run_command_in_background(step['cmd'], step['name']):
            success = False
            pipeline_status['error_message'] = f"Gagal pada langkah: {step['name']}"
            break
            
    pipeline_status['is_running'] = False
    pipeline_status['success'] = success
    
    end_msg = f"\n=== PIPELINE {model_version.upper()} SELESAI ({'SUKSES' if success else 'GAGAL'}) ===\n"
    log_queue.put(end_msg)
    log_history.append(end_msg)
    # Tanda akhir stream logs
    log_queue.put(None)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/history')
def get_history():
    outputs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
    if not os.path.exists(outputs_dir):
        return jsonify([])
        
    # Mengambil folder yang berformat tanggal YYYY-MM-DD
    dirs = []
    for name in os.listdir(outputs_dir):
        path = os.path.join(outputs_dir, name)
        if os.path.isdir(path):
            # Cek format sederhana (misal ada '-' dan panjang 10)
            if len(name) == 10 and name.count('-') == 2:
                dirs.append(name)
                
    # Urutkan tanggal dari terbaru
    dirs.sort(reverse=True)
    return jsonify(dirs)

@app.route('/api/outputs/<date>')
def get_outputs_data(date):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(base_dir, 'outputs', date)
    
    if not os.path.exists(target_dir):
        return jsonify({'error': 'Data tanggal tersebut tidak ditemukan'}), 404
        
    v1_csv_path = os.path.join(target_dir, 'rekomendasi_lengkap_IDX_ALL.csv')
    v3_csv_path = os.path.join(target_dir, 'rekomendasi_lengkap_IDX_ALL_V3.csv')
    
    def parse_recommendation_csv(csv_path):
        if not os.path.exists(csv_path):
            return []
        data = []
        try:
            with open(csv_path, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Memastikan tipe data dikonversi agar mudah di-filter di frontend
                    data.append({
                        'Ticker': row.get('Ticker', ''),
                        'AI_Score': float(row.get('AI_Score', 0)) if row.get('AI_Score') else 0.0,
                        'Harga': float(row.get('Harga', 0)) if row.get('Harga') else 0.0,
                        'MA50': float(row.get('MA50', 0)) if row.get('MA50') else 0.0,
                        'Trx_Miliar': float(row.get('Trx_Miliar', 0)) if row.get('Trx_Miliar') else 0.0
                    })
        except Exception as e:
            print(f"Error parsing CSV {csv_path}: {e}")
        return data

    result = {
        'date': date,
        'has_visual': os.path.exists(os.path.join(target_dir, 'visualisasi_IDX_ALL.png')),
        'has_visual_v3': os.path.exists(os.path.join(target_dir, 'visualisasi_IDX_ALL_V3.png')),
        'recommendations': parse_recommendation_csv(v1_csv_path),
        'recommendations_v3': parse_recommendation_csv(v3_csv_path)
    }
    return jsonify(result)


@app.route('/api/visuals/<date>')
@app.route('/api/visuals/<date>/<version>')
def get_visual_image(date, version='v1'):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_name = 'visualisasi_IDX_ALL.png' if version == 'v1' else 'visualisasi_IDX_ALL_V3.png'
    img_path = os.path.join(base_dir, 'outputs', date, img_name)
    
    if os.path.exists(img_path):
        return send_file(img_path, mimetype='image/png')
    else:
        return jsonify({'error': 'Visualisasi tidak ditemukan'}), 404

@app.route('/api/download/<date>/<version>')
def download_csv(date, version):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    filename = 'rekomendasi_lengkap_IDX_ALL.csv' if version == 'v1' else 'rekomendasi_lengkap_IDX_ALL_V3.csv'
    file_path = os.path.join(base_dir, 'outputs', date, filename)
    
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True, download_name=f"StockMixer_{version.upper()}_{date}.csv")
    else:
        return jsonify({'error': 'File CSV tidak ditemukan'}), 404


@app.route('/api/status')
def get_pipeline_status():
    return jsonify(pipeline_status)

@app.route('/api/run-pipeline', methods=['POST'])
def trigger_pipeline():
    global pipeline_status
    if pipeline_status['is_running']:
        return jsonify({'error': 'Pipeline sedang berjalan, harap tunggu hingga selesai'}), 400
        
    data = request.json or {}
    target_date = data.get('date')
    model_version = data.get('model', 'v1')
    if not target_date:
        target_date = datetime.datetime.now().strftime('%Y-%m-%d')
        
    # Jalankan pipeline dalam background thread
    t = threading.Thread(target=pipeline_thread_worker, args=(target_date, model_version))
    t.daemon = True
    t.start()
    
    return jsonify({'message': f'Pipeline {model_version.upper()} berhasil dipicu', 'date': target_date})

@app.route('/api/pipeline-logs')
def stream_pipeline_logs():
    def generate():
        # Kirim log history pertama kali agar user melihat log yang sudah lewat
        for log_line in log_history:
            yield f"data: {json.dumps({'log': log_line})}\n\n"
            
        if not pipeline_status['is_running'] and not log_history:
            yield f"data: {json.dumps({'log': 'Belum ada aktivitas log.\n'})}\n\n"
            return
            
        while True:
            try:
                # Blokir sebentar menunggu log baru
                line = log_queue.get(timeout=10)
                if line is None:
                    # Akhir dari log stream
                    yield f"data: {json.dumps({'log': '--- END OF LOGS ---\n', 'finished': True})}\n\n"
                    break
                yield f"data: {json.dumps({'log': line})}\n\n"
            except queue.Empty:
                if not pipeline_status['is_running']:
                    break
                # Kirim heartbeat agar koneksi SSE tetap hidup
                yield f"data: {json.dumps({'heartbeat': True})}\n\n"
                
    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    print("=== Memulai StockMixer Web Dashboard Server ===")
    print("Akses UI di http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=True)
