# English-Vietnamese Neural Machine Translation (NMT)

Dự án này tập trung vào việc xây dựng và so sánh các kiến trúc mạng Neural cho bài toán dịch máy (NMT) từ tiếng Anh sang tiếng Việt, sử dụng bộ dữ liệu PhoMT. Đây là nội dung thực hiện cho Lab 4 - Môn học DS201.

## TỔNG QUAN DỰ ÁN

Dự án triển khai 3 biến thể của mô hình Sequence-to-Sequence (Seq2Seq) dựa trên mạng LSTM:
1. Bài 1: Basic Seq2Seq LSTM - Mô hình Encoder-Decoder cơ bản không sử dụng cơ chế chú ý.
2. Bài 2: Bahdanau Attention - Sử dụng cơ chế Additive Attention (theo nghiên cứu của Bahdanau et al., 2014).
3. Bài 3: Luong Attention - Sử dụng cơ chế Multiplicative/Dot Attention (theo nghiên cứu của Luong et al., 2015).

## CẤU TRÚC THƯ MỤC

| Tên File/Thư mục | Mô tả |
| :--- | :--- |
| MODEL-BAHDANAU/ | Lưu trữ checkpoint và log huấn luyện của mô hình Bahdanau Attention. |
| MODEL-BAI1/ | Lưu trữ checkpoint và log của mô hình Seq2Seq cơ bản. |
| MODEL-LUONG/ | Lưu trữ checkpoint và log của mô hình Luong Attention. |
| phomt_dataset.py | Module xử lý Data Loader và tiền xử lý dữ liệu PhoMT. |
| phomt_vocab.py | Module xây dựng và quản lý từ vựng (Vocabulary). |
| seq2seq_basic_lstm.py | Định nghĩa kiến trúc mô hình Seq2Seq LSTM cơ bản. |
| seq2seq_bahdanau_lstm.py | Định nghĩa kiến trúc mô hình với Bahdanau Attention. |
| seq2seq_luong_lstm.py | Định nghĩa kiến trúc mô hình với Luong Attention. |
| train_basic_lstm.py | Script huấn luyện cho mô hình cơ bản. |
| train_bahdanau_lstm.py | Script huấn luyện cho mô hình Bahdanau. |
| train_luong_lstm.py | Script huấn luyện cho mô hình Luong. |

## KẾT QUẢ THỰC NGHIỆM

Kết quả được đánh giá bằng độ đo ROUGE-L trên tập Test của PhoMT:

| Kiến trúc mô hình | ROUGE-L (Best) | Nhận xét |
| :--- | :---: | :--- |
| Basic Seq2Seq | 0.2850 | Gặp hiện tượng lặp từ và nút thắt thông tin. |
| Bahdanau Attention | 0.3324 | Cải thiện khả năng căn chỉnh (alignment) từ vựng. |
| Luong Attention | 0.4451 | Hiệu suất cao nhất, dịch câu dài ổn định và tự nhiên hơn. |

## YÊU CẦU HỆ THỐNG

* Python 3.8+
* PyTorch
* TorchMetrics
* Matplotlib, NumPy, TQDM

## HƯỚNG DẪN CHẠY

1. Chuẩn bị dữ liệu: Đảm bảo các file dữ liệu PhoMT (`small-train.json`, `small-dev.json`, `small-test.json`) nằm đúng thư mục cấu hình trong các file `train_.py`.
2. Huấn luyện mô hình: Chạy script tương ứng với mô hình muốn thử nghiệm. Ví dụ:
   ```bash
   python train_luong_lstm.py

## DOWNLOAD PRE-TRAINED MODEL
Bạn có thể tải các checkpoint đã huấn luyện tại đây:
- [Basic LSTM Model](https://drive.google.com/drive/folders/14ePnCl5fdYxsiAD3kw9MM6H831C-1891?usp=drive_link)
- [Bahdanau Attention Model](https://drive.google.com/drive/folders/1AlG5LpLMyAZD36VdwCn1diEnjAmR1czS?usp=drive_link)
- [Luong Attention Model](https://drive.google.com/drive/folders/1B01Cy6fz8TQK33HSf2M8QKzitrE2d4jm?usp=drive_link)

