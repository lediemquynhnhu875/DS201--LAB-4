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
        # output: (batch_size, src_len, d_model) -> H_s (cho Attention)
        # hidden: (h_n, c_n): (n_encoder, batch_size, d_model) -> Initial Hidden cho Decoder
        output, hidden = self.lstm(embedded)
        
        return output, hidden

# --- 2. BAHDANAU ATTENTION MODULE ---
class BahdanauAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        # W_s * s_i (Trạng thái ẩn Decoder)
        self.W_s = nn.Linear(d_model, d_model, bias=False)
        # W_h * h_j (Các trạng thái ẩn Encoder)
        self.W_h = nn.Linear(d_model, d_model, bias=False)
        # V * tanh(.) (Vector ngữ cảnh)
        self.V = nn.Linear(d_model, 1, bias=False)

    def forward(self, decoder_hidden_state, encoder_outputs):
        # decoder_hidden_state (s_i): (batch_size, 1, d_model) (lớp trên cùng của h_n)
        # encoder_outputs (h_j): (batch_size, src_len, d_model)
        
        src_len = encoder_outputs.shape[1]
        
        # 1. Mở rộng trạng thái decoder để khớp với chiều src_len
        # s_i_expanded: (batch_size, src_len, d_model)
        s_i_expanded = decoder_hidden_state.repeat(1, src_len, 1)

        # 2. Tính Energy Scores (e_ij): V * tanh(W_s * s_i + W_h * h_j)
        # energy: (batch_size, src_len, d_model)
        energy = torch.tanh(self.W_s(s_i_expanded) + self.W_h(encoder_outputs))
        
        # score: (batch_size, src_len, 1)
        score = self.V(energy)
        
        # 3. Tính Attention Weights (alpha_ij)
        # attention_weights: (batch_size, src_len, 1) -> (batch_size, 1, src_len) sau transpose
        attention_weights = F.softmax(score, dim=1).transpose(1, 2)
        
        # 4. Tính Context Vector (c_i): c_i = sum(alpha_ij * h_j)
        # context_vector: (batch_size, 1, d_model)
        context_vector = torch.bmm(attention_weights, encoder_outputs)

        return context_vector, attention_weights

# --- 3. DECODER VỚI BAHDANAU ATTENTION ---
class AttentionDecoderLSTM(nn.Module):
    def __init__(self, vocab_size, d_model, n_decoder, dropout, pad_idx):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_decoder = n_decoder
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        
        # Decoder LSTM nhận đầu vào là [Embedded token || Context Vector]
        # Kích thước đầu vào: d_model (Embedded) + d_model (Context) = 2 * d_model
        self.lstm = nn.LSTM(
            input_size=2 * d_model, 
            hidden_size=d_model,
            num_layers=n_decoder,
            dropout=dropout if n_decoder > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        self.attention = BahdanauAttention(d_model)

        # Layer dự đoán cuối cùng: Input: d_model (h_n) -> Output: vocab_size
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # W_c để tính context vector cho bước 0 (sử dụng W_c * s_0)
        self.W_c = nn.Linear(2 * d_model, d_model) 

    def forward_step(self, input_token, hidden, encoder_outputs):
        # input_token: (batch_size, 1)
        # hidden: (h_n, c_n) từ bước trước, mỗi tensor shape: (n_decoder, batch_size, d_model)
        
        embedded = self.dropout(self.embedding(input_token))
        # embedded: (batch_size, 1, d_model)
        
        # Trạng thái ẩn của lớp trên cùng của LSTM (s_i) cho Bahdanau Attention
        # decoder_hidden_state: (batch_size, 1, d_model)
        # Lấy hidden state của lớp cuối cùng (n_decoder - 1)
        decoder_hidden_state = hidden[0][-1, :, :].unsqueeze(1) 

        # 1. Attention (Tính Context Vector c_i)
        # context: (batch_size, 1, d_model)
        context, attn_weights = self.attention(decoder_hidden_state, encoder_outputs)

        # 2. Concatenation (Embedded và Context)
        # lstm_input: (batch_size, 1, 2*d_model)
        lstm_input = torch.cat((embedded, context), dim=-1)
        
        # 3. LSTM Step
        # output: (batch_size, 1, d_model) -> H_t (Output của LSTM)
        # hidden: (n_decoder, batch_size, d_model) -> State mới cho bước tiếp theo
        output, hidden = self.lstm(lstm_input, hidden)
        
        # 4. Final Prediction (Dự đoán từ)
        # prediction: (batch_size, 1, vocab_size)
        prediction = self.output_layer(output)
        
        return prediction, hidden, attn_weights

    def forward(self, tgt, initial_hidden, encoder_outputs):
        # tgt: (batch_size, tgt_len) - Decoder input (<bos> y1 y2 ...)
        batch_size, tgt_len = tgt.shape
        
        all_predictions = torch.zeros(batch_size, tgt_len, self.vocab_size, device=tgt.device)
        hidden = initial_hidden
        
        for t in range(tgt_len):
            input_token = tgt[:, t].unsqueeze(1) 
            
            prediction, hidden, _ = self.forward_step(input_token, hidden, encoder_outputs)
            
            all_predictions[:, t] = prediction.squeeze(1)
            
        return all_predictions, hidden


class Seq2SeqBahdanauLSTM(nn.Module):
    def __init__(self, d_model, n_encoder, n_decoder, dropout, vocab):
        super().__init__()
        self.vocab = vocab
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.encoder = EncoderLSTM(
            vocab_size=vocab.src_vocab_size,
            d_model=d_model,
            n_encoder=n_encoder,
            dropout=dropout,
            pad_idx=vocab.pad_idx
        )
        
        self.decoder = AttentionDecoderLSTM(
            vocab_size=vocab.tgt_vocab_size,
            d_model=d_model,
            n_decoder=n_decoder,
            dropout=dropout,
            pad_idx=vocab.pad_idx
        )
        
    def forward(self, src, tgt):
        encoder_outputs, encoder_hidden = self.encoder(src)
        # output: (batch_size, tgt_len, vocab_size)
        output, _ = self.decoder(tgt, encoder_hidden, encoder_outputs)
        
        return output
    
    @torch.no_grad()
    def predict(self, src, max_len=50):
        self.eval()
        batch_size = src.shape[0]
        
        encoder_outputs, hidden = self.encoder(src)
        
        input_token = torch.full((batch_size, 1), self.vocab.bos_idx, dtype=torch.long, device=self.device)
        output_tokens = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device)
        output_tokens[:, 0] = self.vocab.bos_idx 
        
        for t in range(1, max_len):
            prediction, hidden, _ = self.decoder.forward_step(input_token, hidden, encoder_outputs)
            
            next_token = prediction.argmax(dim=-1) 
            output_tokens[:, t] = next_token.squeeze(-1)
            
            if ((output_tokens[:, t] == self.vocab.eos_idx) | (output_tokens[:, t] == self.vocab.pad_idx)).all():
                break
                
            input_token = next_token

        return output_tokens
