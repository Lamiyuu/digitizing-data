import os
import time
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer, util

def test_multimodal_split_architecture():
    # Ép buộc hệ thống sử dụng CPU
    device = "cpu"
    
    print("🤖 Đang tải bộ đôi mô hình trích xuất chuyên dụng về CPU...")
    start_init = time.time()
    
    # 1. Mô hình chuyên băm ẢNH (Dùng bản CLIP gốc cực nhẹ của OpenAI)
    # Mô hình này chịu trách nhiệm biến đổi ảnh thành Vector 512 chiều
    image_model = SentenceTransformer('clip-ViT-B-32', device=device)
    
    # 2. Mô hình chuyên băm CHỮ đa ngôn ngữ (Được tối ưu để hiểu tiếng Việt xuất sắc)
    # Mô hình này đã được huấn luyện để đưa text tiếng Việt về ĐÚNG không gian vector 512 chiều của bản CLIP gốc ở trên
    text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1', device=device)
    
    print(f"✅ Tải xong cả 2 mô hình thành công! Thời gian tải: {time.time() - start_init:.2f} giây.")
    print("-" * 60)

    # 3. Kiểm tra và nạp dữ liệu ảnh thô (Ingest Layer)
    img_path1 = "anh_o_to.jpg"
    img_path2 = "anh_xe_may.jpg"

    if not os.path.exists(img_path1) or not os.path.exists(img_path2):
        print(f"❌ Lỗi: Thiếu file ảnh '{img_path1}' hoặc '{img_path2}' trong thư mục hiện tại.")
        return

    image1 = Image.open(img_path1)
    image2 = Image.open(img_path2)

    # 4. Trích xuất Vector Ảnh bằng image_model
    print("⏳ Đang xử lý băm ảnh sang Vector...")
    start_inference = time.time()
    img_vector1 = image_model.encode(image1)
    img_vector2 = image_model.encode(image2)
    
    # 5. Trích xuất Vector Chữ tiếng Việt bằng text_model
    query_text = "Một chiếc xe máy"
    print(f"🔍 Câu lệnh truy vấn tiếng Việt: '{query_text}'")
    text_vector = text_model.encode(query_text)
    
    print(f"✅ Xử lý toán học xong! Kích thước Vector đạt chuẩn: {img_vector1.shape}")
    print(f"⚡ Thời gian CPU xử lý toàn bộ: {time.time() - start_inference:.2f} giây.")
    print("-" * 60)

    # 6. Tính toán độ tương đồng Cosine (Cosine Similarity) giữa Text Vector và các Image Vectors
    score1 = util.cos_sim(text_vector, img_vector1).item()
    score2 = util.cos_sim(text_vector, img_vector2).item()

    print("📝 KẾT QUẢ ĐỐI CHIẾU NGỮ NGHĨA TIẾNG VIỆT CHUẨN:")
    print(f" 🎯 Độ tương đồng với ảnh 1 ({img_path1}): {score1 * 100:.2f}%")
    print(f" 🎯 Độ tương đồng với ảnh 2 ({img_path2}): {score2 * 100:.2f}%")
    print("-" * 60)
    
    if score1 > score2:
        print(f"🏆 Kết quả truy vấn chính xác nhất: {img_path1}")
    else:
        print(f"🏆 Kết quả truy vấn chính xác nhất: {img_path2}")

if __name__ == "__main__":
    test_multimodal_split_architecture()