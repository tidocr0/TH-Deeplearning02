# Bài Tập Thực Hành: Xây Dựng Mô Hình Hồi Quy Tuyến Tính (Linear Regression) với PyTorch

## 📊 Tập dữ liệu (Dataset)
- **File dữ liệu:** `carpricesdata.csv`
- **Biến mục tiêu (Target):** `Price` (Dữ liệu dạng số - Numerical).
- **Biến đầu vào (Input):** Bao gồm cả dữ liệu dạng số (Numerical) và dữ liệu phân loại (Categorical).

## 🛠️ Quy trình thực hiện chi tiết

### 1. Tiền xử lý và Làm sạch dữ liệu (Data Cleaning)
- Sử dụng kĩ thuật `dropna()` trong Pandas để loại bỏ các dòng dữ liệu bị khuyết thiếu (Missing values), đảm bảo tập dữ liệu sạch trước khi đưa vào huấn luyện.
- Phân tách biến mục tiêu (`Price`) và tập đặc trưng (Features). Tự động phân loại các cột thành 2 nhóm: `num_cols` (dạng số) và `cat_cols` (dạng chuỗi).

### 2. Phân chia dữ liệu và Chuẩn hóa (Tránh Data Leakage)
- Tách tập dữ liệu thành 2 phần: **Train** (80%) và **Test** (20%).
- Sử dụng `ColumnTransformer` để thiết lập Pipeline chuẩn hóa:
  - **Dữ liệu số:** Áp dụng `StandardScaler` để đưa về phân phối chuẩn $N(0,1)$.
  - **Dữ liệu chuỗi:** Áp dụng `OneHotEncoder` để chuyển đổi nhị phân.
- **Lưu ý quan trọng:** Để ngăn chặn hoàn toàn hiện tượng rò rỉ dữ liệu (Data Leakage), hệ thống chỉ dùng `fit_transform` trên tập Train và chỉ dùng `transform` trên tập Test.

### 3. Xây dựng và Huấn luyện mô hình
- Chuyển đổi dữ liệu ma trận sang cấu trúc Tensor của PyTorch (`torch.float32`).
- Khởi tạo kiến trúc mạng nơ-ron tuyến tính đơn tầng (`nn.Linear`).
- **Thuật toán tối ưu:** Gradient Descent (`torch.optim.SGD`) với Tốc độ học (Learning Rate) = `0.001`.
- **Hàm mất mát:** Mean Squared Error (`nn.MSELoss`).
- **Chu kỳ huấn luyện:** Thiết lập tối đa `500 epochs`.

### 4. Kỹ thuật Dừng sớm (Early Stopping)
Để khắc phục hiện tượng học vẹt (Overfitting) khi mô hình học quá mức trên tập Train nhưng dự đoán kém trên tập Test, dự án đã tích hợp cơ chế Early Stopping:
- **Monitor:** Theo dõi giá trị Loss trên tập kiểm thử (`val_loss`).
- **Patience:** Thiết lập bằng `10`. Nếu qua 10 epoch liên tiếp mà `val_loss` không giảm, quá trình huấn luyện sẽ lập tức ngắt.
- Tự động khôi phục và lưu lại bộ trọng số (weights) tại epoch đạt kết quả dự đoán tốt nhất.

### 5. Trực quan hóa kết quả
Sử dụng thư viện `matplotlib` để vẽ đồ thị so sánh giữa `Train Loss` và `Validation Loss` theo thời gian thực (Epochs), qua đó đánh giá trực quan điểm tiệm cận và vị trí cực tiểu tối ưu nhất của mô hình.

---

## 🚀 Hướng dẫn chạy mã nguồn (Google Colab)
1. Tải file `carpricesdata.csv` vào thư mục lưu trữ của Google Colab.
2. Sao chép toàn bộ mã nguồn Python.
3. Chạy (Run) ô code. Mô hình sẽ tự động cài đặt thư viện, thực hiện toàn bộ quy trình, in ra log huấn luyện và hiển thị đồ thị đánh giá ở bước cuối cùng.
