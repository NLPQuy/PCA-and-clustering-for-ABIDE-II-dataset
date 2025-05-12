import pandas as pd

# Bước 1: Đọc dữ liệu từ file CSV
df = pd.read_csv('ABIDE2(updated).csv')  # Thay bằng đường dẫn file gốc của bạn

# Bước 2: Lấy ngẫu nhiên 10% dữ liệu
df_sample = df.sample(frac=0.1, random_state=42)  # random_state để tái lập kết quả

# Bước 3: Ghi dữ liệu mẫu xuống file CSV mới
df_sample.to_csv('ABIDE2_sample.csv', index=False)  # index=False để không ghi chỉ mục dòng
