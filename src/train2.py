import random
import numpy as np
import os
import torch as torch
from load_data import load_EOD_data
from evaluator import evaluate
from model import get_loss, StockMixer
import pickle
import warnings

warnings.filterwarnings('ignore')

np.random.seed(123456789)
torch.random.manual_seed(12345678)
device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'

# --- KONFIGURASI YANG HARUS DIUBAH SESUAI DATASET ---
market_name = 'IDX_ALL_V3'    # Ubah ke IDX_ALL_V2 atau IDX_ALL_V3
fea_num = 8                   # Gunakan 8 untuk V2/V3
# ----------------------------------------------------

data_path = '../dataset'
relation_name = 'wikidata'
lookback_length = 16
epochs = 100
valid_index = 1100
test_index = 1350
market_num = 20
steps = 1
learning_rate = 0.001
alpha = 0.1
scale_factor = 3
activation = 'GELU'

dataset_path = '../dataset/' + market_name  # Path sudah disesuaikan untuk run dari luar src
print(f"Memuat dataset dari {dataset_path}...")

# 1. SMART DATA LOADER (Otomatis menyesuaikan dimensi tanpa error transpose)
with open(os.path.join(dataset_path, "eod_data.pkl"), "rb") as f:
    eod_data = pickle.load(f)
    if eod_data.shape[0] > 1000: # Jika array pertama adalah Hari (misal 2051), maka putar!
        eod_data = eod_data.transpose(1, 0, 2)
        
with open(os.path.join(dataset_path, "mask_data.pkl"), "rb") as f:
    mask_data = pickle.load(f)
    if mask_data.shape[0] > 1000:
        mask_data = mask_data.transpose(1, 0)
        
with open(os.path.join(dataset_path, "gt_data.pkl"), "rb") as f:
    gt_data = pickle.load(f)
    if gt_data.shape[0] > 1000:
        gt_data = gt_data.transpose(1, 0)
        
# Anti-crash jika price_data hilang
try:
    with open(os.path.join(dataset_path, "price_data.pkl"), "rb") as f:
        price_data = pickle.load(f)
        if price_data.shape[0] > 1000:
            price_data = price_data.transpose(1, 0)
except FileNotFoundError:
    price_data = np.ones_like(gt_data)

# 2. SETTING DINAMIS: Ambil jumlah saham (stock_num) otomatis dari data
stock_num = eod_data.shape[0]
print(f"[INFO] Data siap! Jumlah Saham terdeteksi: {stock_num} | Fitur: {fea_num}")

trade_dates = mask_data.shape[1]

model = StockMixer(
    stocks=stock_num,
    time_steps=lookback_length,
    channels=fea_num,
    market=market_num,
    scale=scale_factor
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
best_valid_loss = np.inf
best_valid_perf = None
best_test_perf = None
batch_offsets = np.arange(start=0, stop=valid_index, dtype=int)

patience = 5            
patience_counter = 0

def validate(start_index, end_index):
    with torch.no_grad():
        cur_valid_pred = np.zeros([stock_num, end_index - start_index], dtype=float)
        cur_valid_gt = np.zeros([stock_num, end_index - start_index], dtype=float)
        cur_valid_mask = np.zeros([stock_num, end_index - start_index], dtype=float)
        loss, reg_loss, rank_loss = 0., 0., 0.
        
        for cur_offset in range(start_index - lookback_length - steps + 1, end_index - lookback_length - steps + 1):
            data_batch, mask_batch, price_batch, gt_batch = map(
                lambda x: torch.Tensor(x).to(device), get_batch(cur_offset)
            )
            prediction = model(data_batch)
            cur_loss, cur_reg_loss, cur_rank_loss, cur_rr = get_loss(prediction, gt_batch, price_batch, mask_batch, stock_num, alpha)
            
            loss += cur_loss.item()
            reg_loss += cur_reg_loss.item()
            rank_loss += cur_rank_loss.item()
            cur_valid_pred[:, cur_offset - (start_index - lookback_length - steps + 1)] = cur_rr[:, 0].cpu()
            cur_valid_gt[:, cur_offset - (start_index - lookback_length - steps + 1)] = gt_batch[:, 0].cpu()
            cur_valid_mask[:, cur_offset - (start_index - lookback_length - steps + 1)] = mask_batch[:, 0].cpu()
            
        loss /= (end_index - start_index)
        reg_loss /= (end_index - start_index)
        rank_loss /= (end_index - start_index)
        cur_valid_perf = evaluate(cur_valid_pred, cur_valid_gt, cur_valid_mask)
    return loss, reg_loss, rank_loss, cur_valid_perf

def get_batch(offset=None):
    if offset is None:
        offset = random.randrange(0, valid_index)
    seq_len = lookback_length
    mask_batch = np.min(mask_data[:, offset: offset + seq_len + steps], axis=1)
    return (
        eod_data[:, offset:offset + seq_len, :],
        np.expand_dims(mask_batch, axis=1),
        np.expand_dims(price_data[:, offset + seq_len - 1], axis=1),
        np.expand_dims(gt_data[:, offset + seq_len + steps - 1], axis=1)
    )

for epoch in range(epochs):
    print("epoch{}##########################################################".format(epoch + 1))
    np.random.shuffle(batch_offsets)
    tra_loss, tra_reg_loss, tra_rank_loss = 0.0, 0.0, 0.0
    
    for j in range(valid_index - lookback_length - steps + 1):
        data_batch, mask_batch, price_batch, gt_batch = map(
            lambda x: torch.Tensor(x).to(device), get_batch(batch_offsets[j])
        )
        optimizer.zero_grad()
        prediction = model(data_batch)
        cur_loss, cur_reg_loss, cur_rank_loss, _ = get_loss(prediction, gt_batch, price_batch, mask_batch, stock_num, alpha)
        
        cur_loss.backward()
        optimizer.step()

        tra_loss += cur_loss.item()
        tra_reg_loss += cur_reg_loss.item()
        tra_rank_loss += cur_rank_loss.item()
        
    tra_loss /= (valid_index - lookback_length - steps + 1)
    tra_reg_loss /= (valid_index - lookback_length - steps + 1)
    tra_rank_loss /= (valid_index - lookback_length - steps + 1)
    print('Train : loss:{:.2e}  =  {:.2e} + alpha*{:.2e}'.format(tra_loss, tra_reg_loss, tra_rank_loss))

    val_loss, val_reg_loss, val_rank_loss, val_perf = validate(valid_index, test_index)
    print('Valid : loss:{:.2e}  =  {:.2e} + alpha*{:.2e}'.format(val_loss, val_reg_loss, val_rank_loss))

    test_loss, test_reg_loss, test_rank_loss, test_perf = validate(test_index, trade_dates)
    print('Test: loss:{:.2e}  =  {:.2e} + alpha*{:.2e}'.format(test_loss, test_reg_loss, test_rank_loss))

    if val_loss < best_valid_loss:
        best_valid_loss = val_loss
        best_valid_perf = val_perf
        best_test_perf = test_perf
        patience_counter = 0 
        
        os.makedirs('src/models', exist_ok=True)
        model_path = f'src/models/{market_name}_best.pth'
        torch.save(model.state_dict(), model_path)
        print(f'---> Model terbaik disimpan di: {model_path}')
    else:
        patience_counter += 1
        print(f'---> [WARNING] Performa memburuk. Kesabaran: {patience_counter}/{patience}')
        if patience_counter >= patience:
            print(f"\n!!! EARLY STOPPING DIBERLAKUKAN !!!\nTraining dihentikan secara paksa pada Epoch {epoch + 1}.")
            break

    print('Valid performance:\n', 'mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(val_perf['mse'], val_perf['IC'], val_perf['RIC'], val_perf['prec_10'], val_perf['sharpe5']))