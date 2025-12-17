import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

from phomt_dataset import collate_fn, phoMTDataset
from phomt_vocab import Vocab
from seq2seq_basic_lstm import Seq2SeqLSTM

# Import Metrics
try:
    from torchmetrics.text.rouge import ROUGEScore
except ImportError:
    # Trường hợp không cài đặt torchmetrics
    print("Vui lòng cài đặt torchmetrics: pip install torchmetrics")
    exit()

# --- Config ---
device = "cuda" if torch.cuda.is_available() else "cpu"
# CẬP NHẬT ĐƯỜNG DẪN CHECKPOINT
CHECKPOINT_DIR = "/content/drive/MyDrive/DATA-SCIENCE-SUBJECT/DS201/PRACTICE/DS201-LAB4/MODEL-BAI1/"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
DATASET_ROOT = "/content/drive/MyDrive/DATASET/small-PhoMT/" 


# 1. Khởi tạo logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# Xóa các handler cũ nếu có
if logger.hasHandlers():
    logger.handlers.clear()

# 2. Định dạng cho Log File 
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler = logging.FileHandler(os.path.join(CHECKPOINT_DIR, "training.log"), mode='a')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# 3. Định dạng cho Console 
console_formatter = logging.Formatter("%(message)s")
console_handler = logging.StreamHandler()
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)
# --- Kết thúc Logging Setup MỚI ---

def indices_to_text(indices, vocab, is_target=True):
    tokens = []
    i2s = vocab.tgt_i2s if is_target else vocab.src_i2s

    for idx in indices:
        if isinstance(idx, torch.Tensor):
            idx = idx.item()

        if is_target and idx == vocab.eos_idx:
            break

        if idx != vocab.pad_idx:
            if is_target and idx == vocab.bos_idx:
                continue

            token = i2s.get(idx, vocab.unk_token)
            tokens.append(token)

        if not is_target and idx == vocab.eos_idx:
            break

    return " ".join(tokens)

def train(model: nn.Module, dataloader: DataLoader, epoch: int, loss_fn, optimizer):
    model.train()
    running_loss = []

    with tqdm(dataloader, desc=f"Epoch {epoch} - Training") as pbar:
        for item in pbar:
            src = item['src'].to(device)
            tgt = item['tgt'].to(device)

            optimizer.zero_grad()

            decoder_input = tgt[:, :-1] 
            targets = tgt[:, 1:]

            logits = model(src, decoder_input)

            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss.append(loss.item())
            pbar.set_postfix({"loss": np.mean(running_loss)})
            
            if device == "cuda":
                torch.cuda.empty_cache()

    avg_loss = np.mean(running_loss)
    logging.info(f"--- Epoch {epoch} TRAIN finished --- Loss: {avg_loss:.4f}")
    return avg_loss

def evaluate(model: nn.Module, dataloader: DataLoader, epoch: int, loss_fn, vocab):
    model.eval()
    running_loss = []
    rouge_metric = ROUGEScore(rouge_keys=("rougeL",)).to(device) 
    all_preds_text = []
    all_targets_text = []

    example_printed = False

    with torch.no_grad():
        for item in tqdm(dataloader, desc=f"Epoch {epoch} - Evaluating"):
            src = item['src'].to(device)
            tgt = item['tgt'].to(device)

            # 1. Validation Loss (Sử dụng Teacher Forcing)
            decoder_input = tgt[:, :-1]
            targets = tgt[:, 1:]
            logits = model(src, decoder_input)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            running_loss.append(loss.item())

            # 2. ROUGE-L Prediction (Sử dụng Inference)
            generated_tokens = model.predict(src, max_len=tgt.shape[1] + 10) 

            for i in range(len(tgt)):
                pred_seq = generated_tokens[i].tolist()
                pred_text = indices_to_text(pred_seq, vocab, is_target=True)

                tgt_seq = tgt[i].tolist()
                tgt_text = indices_to_text(tgt_seq, vocab, is_target=True)

                # LOGIC IN VÍ DỤ 
                if not example_printed:
                    src_seq = src[i].tolist()
                    src_text = indices_to_text(src_seq, vocab, is_target=False)

                    logging.info(f"\n======== Example Translation (Epoch {epoch}) ========")
                    logging.info(f"-> Source (EN):     {src_text}")
                    logging.info(f"-> Reference (VN):  {tgt_text}")
                    logging.info(f"-> Prediction (VN): {pred_text}")
                    logging.info("==================================================")
                    example_printed = True 

                all_preds_text.append(pred_text)
                all_targets_text.append(tgt_text)
            
            if device == "cuda":
                torch.cuda.empty_cache()

    # Tính ROUGE trên tập Validation
    if len(all_preds_text) > 0:
        rouge_scores = rouge_metric(all_preds_text, all_targets_text)
        rouge_l = rouge_scores['rougeL_fmeasure'].item()
    else:
        rouge_l = 0.0

    avg_loss = np.mean(running_loss)
    logging.info(f"--- Epoch {epoch} EVAL finished --- Val Loss: {avg_loss:.4f} | ROUGE-L: {rouge_l:.4f}")

    return avg_loss, rouge_l

def visualize_metrics(train_losses, val_losses, rouge_scores):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Val Loss', marker='s')
    plt.title("Loss History")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, rouge_scores, label='Val ROUGE-L', marker='^', color='green')
    plt.title("ROUGE-L Score History")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def main():
    logging.info("="*50)
    logging.info(f"Starting training on Device: {device}")
    
    vocab = Vocab(
        path=DATASET_ROOT,
        src_language="english",
        tgt_language="vietnamese"
    )

    # ĐỌC DATASET
    train_dataset = phoMTDataset(os.path.join(DATASET_ROOT, "small-train.json"), vocab)
    dev_dataset = phoMTDataset(os.path.join(DATASET_ROOT, "small-dev.json"), vocab)
    test_dataset = phoMTDataset(os.path.join(DATASET_ROOT, "small-test.json"), vocab)
    
    logging.info(f"Using full datasets: Train size={len(train_dataset)}, Dev size={len(dev_dataset)}, Test size={len(test_dataset)}")
    
    BATCH_SIZE = 16 

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Khởi tạo model với thông số yêu cầu
    model = Seq2SeqLSTM(
        d_model=256, # Hidden Size 256
        n_encoder=3, # 3 lớp Encoder
        n_decoder=3, # 3 lớp Decoder
        dropout=0.3,
        vocab=vocab
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model Parameters (LSTM Unidirectional): {total_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx) 

    # Đổi tên checkpoint file
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_basic_lstm_mt.pt") 
    best_rouge = 0.0
    start_epoch = 0

    # Tải Checkpoint nếu tồn tại
    if os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt['epoch']
            best_rouge = ckpt.get('best_rouge', 0.0)
            logging.info(f"Resumed from epoch {start_epoch}, Best ROUGE: {best_rouge:.4f}")
        except Exception as e:
            logging.warning(f"Error loading checkpoint: {e}. Starting from scratch.")

    train_losses, val_losses, val_rouges = [], [], []
    patience = 0

    for epoch in range(start_epoch + 1, 20):
        logging.info(f"\n--- Starting Epoch {epoch} ---")

        t_loss = train(model, train_loader, epoch, loss_fn, optimizer)
        v_loss, v_rouge = evaluate(model, dev_loader, epoch, loss_fn, vocab)

        train_losses.append(t_loss)
        val_losses.append(v_loss)
        val_rouges.append(v_rouge)

        if v_rouge > best_rouge:
            best_rouge = v_rouge
            patience = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_rouge': best_rouge
            }, checkpoint_path)
            # THAY ĐỔI THÔNG BÁO SAVE CHECKPOINT
            logging.info(f"!!! NEW BEST MODEL (Epoch {epoch}) !!! Saved ROUGE-L: {best_rouge:.4f}")
        else:
            patience += 1
            logging.info(f"No improvement. Patience: {patience}/10")
            if patience >= 10:
                logging.info("Early stopping!")
                break

    logging.info("\n================= Final Test Evaluation =================")
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])

        test_loss, test_rouge = evaluate(model, test_loader, 0, loss_fn, vocab)
        logging.info(f"Final Test Loss: {test_loss:.4f} | Test ROUGE-L: {test_rouge:.4f}")
    else:
        logging.warning("Cannot evaluate on Test Set: Best model checkpoint not found.")

    visualize_metrics(train_losses, val_losses, val_rouges)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Sử dụng logger để ghi lỗi vào file
        logger.error(f"Error: {str(e)}", exc_info=True)
        # In thông báo lỗi ngắn gọn ra console
        print(f"\n[FATAL ERROR] Check training.log for details. Error: {str(e)}")
