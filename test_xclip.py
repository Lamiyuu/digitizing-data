import os
import cv2
import numpy as np
import torch
import faiss
from decord import VideoReader, cpu
from transformers import XCLIPModel, AutoTokenizer

# Cấu hình thiết bị chạy AI (Sử dụng GPU nếu có, không thì CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- HÀM TIỀN XỬ LÝ VIDEO ĐỘC LẬP ---
# (Thay thế hoàn toàn bộ XCLIPProcessor bị lỗi trên Windows)
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

def manual_preprocess_video(video_path, num_frames=8):
    # Đọc video và trích xuất số khung hình cách đều nhau
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    raw_frames = vr.get_batch(indices).asnumpy()

    processed_frames = []
    for frame in raw_frames:
        # Resize về kích thước chuẩn của mạng X-CLIP
        frame_resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_CUBIC)
        # Chuyển đổi định dạng màu
        frame_float = frame_resized.astype(np.float32) / 255.0
        frame_norm = (frame_float - CLIP_MEAN) / CLIP_STD
        # PyTorch yêu cầu định dạng (Kênh màu, Chiều cao, Chiều rộng)
        frame_transposed = np.transpose(frame_norm, (2, 0, 1))
        processed_frames.append(frame_transposed)

    # Đóng gói thành Tensor hoàn chỉnh (Batch=1, Frames=8, C=3, H=224, W=224)
    video_tensor = np.stack(processed_frames)
    video_tensor = np.expand_dims(video_tensor, axis=0)
    
    return torch.tensor(video_tensor).to(device)


class VideoRetrievalSystem:
    def __init__(self):
        print("Đang tải mạng lõi X-CLIP và hệ thống FAISS...")
        self.model_name = "microsoft/xclip-base-patch16"
        
        # Chỉ tải Tokenizer cho phần chữ và Mạng lõi cho phần ảnh
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = XCLIPModel.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to(device)
        
        # Khởi tạo FAISS Index (Dùng khoảng cách Cosine)
        self.dimension = 512 
        self.index = faiss.IndexFlatIP(self.dimension) 
        
        # Lưu trữ ánh xạ giữa ID và tên file video
        self.video_database = {}
        self.video_counter = 0

    def add_video_to_db(self, video_path):
        print(f"Đang trích xuất Vector từ: {video_path}")
        pixel_values = manual_preprocess_video(video_path, num_frames=8)

        with torch.no_grad():
            video_outputs = self.model.get_video_features(pixel_values=pixel_values)
            
            # --- BÓC HỘP TENSOR THÔNG MINH ---
            if isinstance(video_outputs, torch.Tensor):
                video_features = video_outputs
            elif hasattr(video_outputs, "video_embeds"):
                video_features = video_outputs.video_embeds
            elif hasattr(video_outputs, "pooler_output"):
                video_features = video_outputs.pooler_output
            else:
                video_features = video_outputs[0]
            
            # Chuẩn hóa L2 (Ép độ dài vector về 1 để so sánh góc Cosine)
            video_features = video_features / video_features.norm(dim=-1, keepdim=True)
            embedding = video_features.squeeze().cpu().numpy().astype("float32")

        # Đẩy vector vào cơ sở dữ liệu FAISS
        self.index.add(np.array([embedding]))
        self.video_database[self.video_counter] = video_path
        self.video_counter += 1
        print(" -> Đã thêm vào Database thành công!")

    def search_video_by_text(self, text_query, top_k=1):
        print(f"\n[TÌM KIẾM]: '{text_query}'")
        
        # Tiền xử lý chữ
        inputs = self.tokenizer(text_query, padding=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            text_outputs = self.model.get_text_features(**inputs)
            
            # --- BÓC HỘP TENSOR THÔNG MINH ---
            if isinstance(text_outputs, torch.Tensor):
                text_features = text_outputs
            elif hasattr(text_outputs, "text_embeds"):
                text_features = text_outputs.text_embeds
            elif hasattr(text_outputs, "pooler_output"):
                text_features = text_outputs.pooler_output
            else:
                text_features = text_outputs[0]
            
            # Chuẩn hóa L2 Text Vector
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_embedding = text_features.cpu().numpy().astype("float32")

        # FAISS tìm kiếm vector video có khoảng cách gần nhất
        similarities, indices = self.index.search(text_embedding, top_k)

        # Hiển thị kết quả
        print("Kết quả khớp nhất:")
        for i in range(top_k):
            video_id = indices[0][i]
            score = similarities[0][i]
            
            # FAISS trả về ID -1 nếu Database trống hoặc có lỗi
            if video_id != -1:
                print(f" - Hạng {i+1}: {self.video_database[video_id]} (Độ tự tin: {score:.4f})")
            else:
                print(" - Không có dữ liệu.")


# ==========================================
# KHU VỰC CHẠY KIỂM THỬ THỰC TẾ
# ==========================================
if __name__ == "__main__":
    
    # 1. Khởi động AI
    search_engine = VideoRetrievalSystem()
    
    # 2. Bạn hãy đảm bảo file mp4 này đang nằm chung thư mục với file code
    video_1 = "dog.mp4" 
    
    if os.path.exists(video_1):
        # Đưa video vào hệ thống
        search_engine.add_video_to_db(video_1)
        
        # Bạn có thể gọi thêm hàm add_video_to_db() nếu có nhiều video khác
        # search_engine.add_video_to_db("video_khac.mp4")
        
        # 3. Test tìm kiếm bằng các câu lệnh Tiếng Anh khác nhau
        search_engine.search_video_by_text("a person", top_k=1)
        search_engine.search_video_by_text("a dog", top_k=1)
        search_engine.search_video_by_text("a cat", top_k=1)
        
    else:
        print(f"\n[LỖI]: Không tìm thấy file '{video_1}'.")
        print("Bạn vui lòng kiểm tra lại tên file video và để nó cùng chỗ với file code nhé!")