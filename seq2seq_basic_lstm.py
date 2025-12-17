import torch
from torch import nn
import torch.nn.functional as F

class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size, d_model, n_encoder, dropout, pad_idx):
        super().__init__()
        self.d_model = d_model
        self.n_encoder = n_encoder
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        
        # LSTM Unidirectional: num_layers=n_encoder, batch_first=True
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_encoder,
            dropout=dropout if n_encoder > 1 else 0,
            batch_first=True,
            bidirectional=False # Chỉ sử dụng Unidirectional
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: (batch_size, src_len)
        embedded = self.dropout(self.embedding(src))
        # embedded: (batch_size, src_len, d_model)
        
        # output: (batch_size, src_len, d_model * num_directions)
        # (h_n, c_n): (num_layers * num_directions, batch_size, d_model)
        output, (h_n, c_n) = self.lstm(embedded)
        
        # Do là Unidirectional, h_n và c_n đã có shape đúng: (n_encoder, batch_size, d_model)
        return output, (h_n, c_n)

class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, d_model, n_decoder, dropout, pad_idx):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_decoder = n_decoder
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_decoder,
            dropout=dropout if n_decoder > 1 else 0,
            batch_first=True,
            bidirectional=False # Chỉ sử dụng Unidirectional
        )
        
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, encoder_hidden):
        # tgt: (batch_size, tgt_len) - Decoder input (ví dụ: <bos> y1 y2 ...)
        # encoder_hidden: (h_n, c_n) từ Encoder, shape: (n_decoder, batch_size, d_model)
        
        embedded = self.dropout(self.embedding(tgt))
        # embedded: (batch_size, tgt_len, d_model)
        
        # output: (batch_size, tgt_len, d_model)
        output, hidden = self.lstm(embedded, encoder_hidden)
        
        # output_layer: (batch_size, tgt_len, vocab_size)
        prediction = self.output_layer(output)
        
        return prediction, hidden

class Seq2SeqLSTM(nn.Module):
    def __init__(self, d_model, n_encoder, n_decoder, dropout, vocab):
        super().__init__()
        self.vocab = vocab
        
        self.encoder = EncoderLSTM(
            vocab_size=vocab.src_vocab_size,
            d_model=d_model,
            n_encoder=n_encoder,
            dropout=dropout,
            pad_idx=vocab.pad_idx
        )
        
        self.decoder = DecoderLSTM(
            vocab_size=vocab.tgt_vocab_size,
            d_model=d_model,
            n_decoder=n_decoder,
            dropout=dropout,
            pad_idx=vocab.pad_idx
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(self, src, tgt):
        # src: (batch_size, src_len)
        # tgt: (batch_size, tgt_len)
        
        _, encoder_hidden = self.encoder(src)
        # encoder_hidden là (h_n, c_n) từ Encoder: (n_layers, batch_size, d_model)
        
        # Decoder sử dụng trạng thái cuối cùng của Encoder làm trạng thái ẩn ban đầu
        output, _ = self.decoder(tgt, encoder_hidden)
        # output: (batch_size, tgt_len, vocab_size)
        
        return output
    
    @torch.no_grad()
    def predict(self, src, max_len=50):
        self.eval()
        batch_size = src.shape[0]
        
        # 1. Encoding
        _, hidden = self.encoder(src)
        
        # 2. Decoding - Bắt đầu với <bos> token
        # input_token: (batch_size, 1) chứa <bos>
        input_token = torch.full((batch_size, 1), self.vocab.bos_idx, dtype=torch.long, device=self.device)
        
        # Tensor để lưu trữ các token được sinh ra
        output_tokens = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device)
        output_tokens[:, 0] = self.vocab.bos_idx # Giữ lại <bos> ở vị trí đầu tiên
        
        for t in range(1, max_len):
            # output: (batch_size, 1, vocab_size), hidden: state mới
            output, hidden = self.decoder(input_token, hidden)
            
            # Lấy token có xác suất cao nhất: next_token (batch_size, 1)
            next_token = output.argmax(dim=-1) 
            
            # Lưu token vào tensor kết quả
            output_tokens[:, t] = next_token.squeeze(-1)
            
            # Dừng nếu tất cả các câu trong batch đã sinh ra <eos>
            if ((output_tokens[:, t] == self.vocab.eos_idx) | (output_tokens[:, t] == self.vocab.pad_idx)).all():
                break
                
            # Cập nhật input cho bước thời gian tiếp theo
            input_token = next_token # (batch_size, 1)

        return output_tokens
