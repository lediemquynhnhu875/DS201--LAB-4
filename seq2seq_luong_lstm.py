import torch
from torch import nn
import torch.nn.functional as F

# --- 1. ENCODER ---
class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size, d_model, n_encoder, dropout, pad_idx):
        super().__init__()
        self.d_model = d_model
        self.n_encoder = n_encoder
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

        # LSTM Unidirectional
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_encoder,
            dropout=dropout if n_encoder > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: (batch_size, src_len)
        embedded = self.dropout(self.embedding(src))
        # embedded: (batch_size, src_len, d_model)

        # output: (batch_size, src_len, d_model) -> output này chính là H_s dùng trong Attention
        # hidden: (h_n, c_n): (n_encoder, batch_size, d_model)
        output, hidden = self.lstm(embedded)

        return output, hidden

# --- 2. LUONG ATTENTION MODULE ---
class LuongAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # Luong Attention sử dụng cơ chế "Dot Product" nếu d_model bằng nhau
        self.d_model = d_model

    def forward(self, decoder_output, encoder_outputs):
        # decoder_output (Q): H_t (trạng thái ẩn hiện tại của decoder) - (batch_size, 1, d_model)
        # encoder_outputs (K): H_s (các trạng thái ẩn của encoder) - (batch_size, src_len, d_model)

        # Tính điểm năng lượng (energy scores): score(H_t, H_s)
        # Luong 'dot' score: H_t * H_s^T
        # scores: (batch_size, 1, src_len)
        scores = torch.bmm(decoder_output, encoder_outputs.transpose(1, 2))

        # Tính trọng số Attention: attention_weights
        # attention_weights: (batch_size, 1, src_len) -> tổng theo chiều src_len = 1
        attention_weights = F.softmax(scores, dim=-1)

        # Tính Context Vector (C_t): Context = attention_weights * H_s
        # context_vector: (batch_size, 1, d_model)
        context_vector = torch.bmm(attention_weights, encoder_outputs)

        return context_vector, attention_weights

# --- 3. DECODER VỚI ATTENTION ---
class AttentionDecoderLSTM(nn.Module):
    def __init__(self, vocab_size, d_model, n_decoder, dropout, pad_idx):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_decoder = n_decoder

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

        # LSTM nhận đầu vào là (embedded + context) nhưng ở đây ta dùng embedded (d_model)
        # và kết hợp context sau bước LSTM (Luông)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_decoder,
            dropout=dropout if n_decoder > 1 else 0,
            batch_first=True,
            bidirectional=False
        )

        self.attention = LuongAttention(d_model)

        # Layer kết hợp output của LSTM và Context vector
        # Input size: 2 * d_model (h_t + C_t)
        self.concat_layer = nn.Linear(2 * d_model, d_model)

        # Layer dự đoán cuối cùng
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward_step(self, input_token, hidden, encoder_outputs):
        # input_token: (batch_size, 1)
        # hidden: (h_n, c_n) từ bước trước, mỗi tensor shape: (n_decoder, batch_size, d_model)
        # encoder_outputs: (batch_size, src_len, d_model)

        embedded = self.dropout(self.embedding(input_token))
        # embedded: (batch_size, 1, d_model)

        # 1. LSTM Step
        # output: (batch_size, 1, d_model) -> H_t
        # hidden: (n_decoder, batch_size, d_model) -> H_n, C_n cho bước tiếp theo
        output, hidden = self.lstm(embedded, hidden)

        # 2. Attention
        # context: (batch_size, 1, d_model)
        # attn_weights: (batch_size, 1, src_len)
        context, attn_weights = self.attention(output, encoder_outputs)

        # 3. Concatenation (H_t và C_t) và Projection
        # Concatenated: (batch_size, 1, 2*d_model)
        concat_input = torch.cat((output, context), dim=-1)

        # Concat -> Tanh -> Projected: (batch_size, 1, d_model)
        output_att = torch.tanh(self.concat_layer(concat_input))

        # 4. Final Prediction Layer
        # prediction: (batch_size, 1, vocab_size)
        prediction = self.output_layer(output_att)

        return prediction, hidden, attn_weights

    def forward(self, tgt, initial_hidden, encoder_outputs):
        # tgt: (batch_size, tgt_len) - Decoder input (<bos> y1 y2 ...)
        batch_size, tgt_len = tgt.shape

        # tensor lưu trữ logits dự đoán
        all_predictions = torch.zeros(batch_size, tgt_len, self.vocab_size, device=tgt.device)

        # Hidden state ban đầu lấy từ Encoder
        hidden = initial_hidden

        # Chạy Decoder theo từng bước thời gian (teacher forcing)
        for t in range(tgt_len):
            # input_token: (batch_size, 1)
            input_token = tgt[:, t].unsqueeze(1)

            # prediction: (batch_size, 1, vocab_size)
            prediction, hidden, _ = self.forward_step(input_token, hidden, encoder_outputs)

            # Lưu trữ prediction
            all_predictions[:, t] = prediction.squeeze(1)

        return all_predictions, hidden


class Seq2SeqAttentionLSTM(nn.Module):
    def __init__(self, d_model, n_encoder, n_decoder, dropout, vocab):
        super().__init__()
        self.vocab = vocab
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Encoder (không đổi)
        self.encoder = EncoderLSTM(
            vocab_size=vocab.src_vocab_size,
            d_model=d_model,
            n_encoder=n_encoder,
            dropout=dropout,
            pad_idx=vocab.pad_idx
        )

        # Decoder (Attention)
        self.decoder = AttentionDecoderLSTM(
            vocab_size=vocab.tgt_vocab_size,
            d_model=d_model,
            n_decoder=n_decoder,
            dropout=dropout,
            pad_idx=vocab.pad_idx
        )

    def forward(self, src, tgt):
        # src: (batch_size, src_len)
        # tgt: (batch_size, tgt_len)

        # encoder_outputs: (batch_size, src_len, d_model) - Dùng cho Attention
        # encoder_hidden: (h_n, c_n): (n_layers, batch_size, d_model) - Dùng làm initial hidden
        encoder_outputs, encoder_hidden = self.encoder(src)

        # output: (batch_size, tgt_len, vocab_size)
        output, _ = self.decoder(tgt, encoder_hidden, encoder_outputs)

        return output

    @torch.no_grad()
    def predict(self, src, max_len=50):
        self.eval()
        batch_size = src.shape[0]

        # 1. Encoding
        encoder_outputs, hidden = self.encoder(src)

        # 2. Decoding - Bắt đầu với <bos> token
        input_token = torch.full((batch_size, 1), self.vocab.bos_idx, dtype=torch.long, device=self.device)

        output_tokens = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device)
        output_tokens[:, 0] = self.vocab.bos_idx

        for t in range(1, max_len):
            # prediction: (batch_size, 1, vocab_size)
            # hidden: state mới
            prediction, hidden, _ = self.decoder.forward_step(input_token, hidden, encoder_outputs)

            # Lấy token có xác suất cao nhất: next_token (batch_size, 1)
            next_token = prediction.argmax(dim=-1)

            # Lưu token vào tensor kết quả
            output_tokens[:, t] = next_token.squeeze(-1)

            # Dừng nếu tất cả các câu trong batch đã sinh ra <eos>
            if ((output_tokens[:, t] == self.vocab.eos_idx) | (output_tokens[:, t] == self.vocab.pad_idx)).all():
                break

            # Cập nhật input cho bước thời gian tiếp theo
            input_token = next_token # (batch_size, 1)

        return output_tokens
