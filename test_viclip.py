import os
import cv2
import numpy as np
import torch
import faiss
from decord import VideoReader, cpu
from transformers import AutoModel, AutoTokenizer

# --- FIX LỖI CODE CỦA TÁC GIẢ (Thiếu file từ điển) ---
from huggingface_hub import hf_hub_download

vocab_file = "bpe_simple_vocab_16e6.txt.gz"
if not os.path.exists(vocab_file):
    print("[HỆ THỐNG] Đang tải file từ điển (vocab) bị thiếu của tác giả...")
    hf_hub_download(repo_id="OpenGVLab/ViCLIP-B-16-hf", filename=vocab_file, local_dir=".")
# ----------------------------------------------------

# Cấu hình thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- HÀM TIỀN XỬ LÝ VIDEO ĐỘC LẬP (Lách lỗi Windows chặn Pillow) ---
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

def manual_preprocess_video(video_path, num_frames=8):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    raw_frames = vr.get_batch(indices).asnumpy()

    processed_frames = []
    for frame in raw_frames:
        frame_resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_CUBIC)
        frame_float = frame_resized.astype(np.float32) / 255.0
        frame_norm = (frame_float - CLIP_MEAN) / CLIP_STD
        frame_transposed = np.transpose(frame_norm, (2, 0, 1))
        processed_frames.append(frame_transposed)

    video_tensor = np.stack(processed_frames)
    video_tensor = np.expand_dims(video_tensor, axis=0)
    return torch.tensor(video_tensor).to(device)


class ViCLIPRetrievalSystem:
    def __init__(self):
        print("\n[HỆ THỐNG] Đang tải mạng lõi ViCLIP (InternVideo)...")
        self.model_name = "OpenGVLab/ViCLIP-B-16-hf"
        
        # Tải mô hình với quyền cho phép chạy mã nguồn tùy chỉnh
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
        self.model.eval()
        self.model.to(device)
        
        # Xử lý Tokenizer linh hoạt
        if hasattr(self.model, 'tokenizer'):
            self.tokenizer = self.model.tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            
        # Khởi tạo cơ sở dữ liệu FAISS
        self.dimension = 512 
        self.index = faiss.IndexFlatIP(self.dimension) 
        self.video_database = {}
        self.video_counter = 0

    def get_video_vector(self, pixel_values):
        # Bộ quét thông minh tìm hàm lõi
        if hasattr(self.model, 'get_vid_features'):
            return self.model.get_vid_features(pixel_values)
        elif hasattr(self.model, 'encode_vision'):
            return self.model.encode_vision(pixel_values)
        elif hasattr(self.model, 'encode_video'):
            return self.model.encode_video(pixel_values)
        else:
            raise AttributeError("Không tìm thấy hàm rút trích vector Video từ mã nguồn gốc.")

    def get_text_vector(self, text_query):
        with torch.no_grad():
            # ĐƯA THẲNG CHUỖI CHỮ VÀO, KHÔNG BỌC LIST NỮA
            if hasattr(self.model, 'get_text_features'):
                return self.model.get_text_features(text_query, self.tokenizer)
            elif hasattr(self.model, 'encode_text'):
                return self.model.encode_text(text_query, self.tokenizer)
            else:
                raise AttributeError("Không tìm thấy hàm rút trích vector Text từ ViCLIP.")

    def add_video_to_db(self, video_path):
        print(f"[FAISS] Đang xử lý Video: {video_path}")
        pixel_values = manual_preprocess_video(video_path, num_frames=8)

        with torch.no_grad():
            video_features = self.get_video_vector(pixel_values)
            
            # Chuẩn hóa L2 (Ép độ dài vector về 1)
            video_features = video_features / video_features.norm(dim=-1, keepdim=True)
            embedding = video_features.squeeze().cpu().numpy().astype("float32")

        # Lưu vào FAISS
        self.index.add(np.array([embedding]))
        self.video_database[self.video_counter] = video_path
        self.video_counter += 1
        print(" -> Đã lưu Vector vào Database thành công!")

    def search_video_by_text(self, text_query, top_k=1):
        print(f"\n[TÌM KIẾM]: '{text_query}'")
        
        with torch.no_grad():
            text_features = self.get_text_vector(text_query)
            
            # Chuẩn hóa L2
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_embedding = text_features.cpu().numpy().astype("float32")

        # Tìm kiếm bằng Cosine Similarity
        similarities, indices = self.index.search(text_embedding, top_k)

        print("Kết quả khớp nhất:")
        for i in range(top_k):
            video_id = indices[0][i]
            score = similarities[0][i]
            if video_id != -1:
                print(f" - Hạng {i+1}: {self.video_database[video_id]} (Độ tự tin: {score:.4f})")
            else:
                print(" - Không có dữ liệu phù hợp.")


# ==========================================
# KHU VỰC CHẠY KIỂM THỬ THỰC TẾ
# ==========================================
if __name__ == "__main__":
    
    # 1. Khởi động hệ thống
    search_engine = ViCLIPRetrievalSystem()
    
    # 2. Bạn hãy đảm bảo file mp4 này đang nằm chung thư mục với file code
    video_1 = "car.mp4" 
    
    if os.path.exists(video_1):
        # Đưa video vào hệ thống
        search_engine.add_video_to_db(video_1)
        
        # 3. Test tìm kiếm bằng các câu lệnh Tiếng Anh khác nhau
        search_engine.search_video_by_text("car", top_k=1)
        search_engine.search_video_by_text("dog", top_k=1)
        search_engine.search_video_by_text("plane", top_k=1)
    else:
        print(f"\n[LỖI]: Không tìm thấy file '{video_1}'.")
        print("Bạn vui lòng kiểm tra lại tên file video và để nó cùng chỗ với file code nhé!")