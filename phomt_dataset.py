import json
import torch
from torch.utils.data import Dataset, DataLoader
import os
import sys

# Đảm bảo có thể import Vocab nếu file này được chạy độc lập
if __name__ != "__main__":
    from phomt_vocab import Vocab 
else:
    # Thêm đường dẫn dự án để import được Vocab khi chạy file này độc lập
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        from phomt_vocab import Vocab
    except ImportError:
        print("Không tìm thấy phomt_vocab.py. Đảm bảo nó cùng thư mục.")
        exit()


class phoMTDataset(Dataset):
    def __init__(self, data_path, vocab):
        self.vocab = vocab
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        src_text = item[self.vocab.src_language]
        tgt_text = item[self.vocab.tgt_language]
        
        # Chuyển đổi thành indices
        src_indices = self.vocab.encode(src_text, is_target=False)
        tgt_indices = self.vocab.encode(tgt_text, is_target=True)

        return {
            'src': torch.tensor(src_indices, dtype=torch.long),
            'tgt': torch.tensor(tgt_indices, dtype=torch.long)
        }

def collate_fn(batch):
    # Tìm chiều dài lớn nhất của câu trong batch
    src_lens = [len(item['src']) for item in batch]
    tgt_lens = [len(item['tgt']) for item in batch]
    max_src_len = max(src_lens)
    max_tgt_len = max(tgt_lens)
    
    # Lấy pad index (giả định pad_idx là 0)
    pad_idx = 0 

    # Padding
    padded_src = torch.full((len(batch), max_src_len), pad_idx, dtype=torch.long)
    padded_tgt = torch.full((len(batch), max_tgt_len), pad_idx, dtype=torch.long)
    
    for i, item in enumerate(batch):
        padded_src[i, :src_lens[i]] = item['src']
        padded_tgt[i, :tgt_lens[i]] = item['tgt']

    return {
        'src': padded_src,
        'tgt': padded_tgt
    }
