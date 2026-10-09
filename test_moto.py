import os
import time
import torch
import open_clip
from PIL import Image

def test_eva_clip_on_cpu():
    # 1. Cấu hình thiết bị chạy cứng là CPU
    device = torch.device("cpu")
    
    print("🤖 Đang khởi tạo mô hình EVA-CLIP (Bản thế hệ 2: EVA02-B-16)...")
    start_init = time.time()
    
    # create_model_and_transforms tự động tải kiến trúc EVA02 bản Base (khoảng hơn 100M params)
    # Tập trọng số 'merged2b' giúp mô hình đạt độ chính xác tối ưu trên CPU
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name='EVA02-B-16',
        pretrained='merged2b_s8b_b131k',
        device=device
    )
    
    # Khởi tạo bộ băm từ ngữ (Tokenizer) tương ứng của EVA02
    tokenizer = open_clip.get_tokenizer('EVA02-B-16')
    
    print(f"✅ Tải mô hình thành công! Thời gian nạp: {time.time() - start_init:.2f} giây.")
    print("-" * 60)

    # 2. Đọc và tiền xử lý ảnh thô bằng PIL & Hàm biến đổi của EVA
    img_path = "anh_xe_may.jpg"
    if not os.path.exists(img_path):
        print(f"❌ Không tìm thấy file ảnh '{img_path}'. Hãy chuẩn bị ảnh để test.")
        return
        
    image_raw = Image.open(img_path)
    
    # Hàm preprocess sẽ tự động resize, crop và normalize ảnh về chuẩn của mạng EVA
    # unsqueeze(0) dùng để thêm chiều batch size -> [1, 3, 224, 224] giống như bạn đã debug
    image_input = preprocess(image_raw).unsqueeze(0).to(device)

    # 3. Chuẩn bị câu truy vấn chữ 
    # Lưu ý: EVA-CLIP nguyên bản hiểu sâu nhất bằng tiếng Anh
    query_texts = ["a photo of a motorcycle", "a photo of a car", "a photo of a cat"]
    text_input = tokenizer(query_texts).to(device)

    # 4. Tiến hành trích xuất đặc trưng (Inference) trên CPU
    print("⏳ EVA-CLIP đang tiến hành băm dữ liệu sang Vector trên CPU...")
    start_inference = time.time()
    
    # Tắt tính năng tính đạo hàm (no_grad) để CPU chạy nhanh hơn và tiết kiệm RAM
    with torch.no_grad():
        # Trích xuất vector ảnh và vector chữ
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)
        
        # CHUẨN HÓA TOÁN HỌC (L2 Normalization): 
        # Bắt buộc phải chia cho norm để đưa vector về độ dài đơn vị trước khi tính Cosine
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Tính toán ma trận độ tương đồng Cosine trực tiếp bằng phép nhân ma trận (@)
        # Nhân thêm hệ số logit_scale (thường quanh mức 100) để kéo giãn biên độ điểm số
        logit_scale = model.logit_scale.exp()
        similarity_scores = (logit_scale * image_features @ text_features.T).softmax(dim=-1)

    print(f"✅ Xử lý xong! Thời gian CPU thực thi: {time.time() - start_inference:.2f} giây.")
    print("-" * 60)

    # 5. In bảng thống kê kết quả phân loại (Xác suất % phân phối qua hàm Softmax)
    print("📊 THỐNG KÊ XÁC SUẤT PHÂN LOẠI CỦA EVA-CLIP:")
    probabilities = similarity_scores[0].tolist()
    
    for i, text in enumerate(query_texts):
        print(f" 🎯 Câu lệnh: '{text:<25}' | Xác suất khớp: {probabilities[i] * 100:.2f}%")
    print("-" * 60)

if __name__ == "__main__":
    test_eva_clip_on_cpu()