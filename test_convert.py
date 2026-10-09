import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

def debug_clip_vision_steps(image_path):
    print("🤖 Đang tải mô hình lõi CLIP-ViT-B-32 để Debug...")
    # Tải Processor (chịu trách nhiệm Tiền xử lý & Cắt Patch)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    # Tải Model lõi và bật chế độ xuất các Trạng thái ẩn (output_hidden_states=True)
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.eval() # Đưa mô hình về chế độ suy luận
    
    # ---------------------------------------------------------
    # BƯỚC 1: Đọc ảnh và Tiền xử lý
    # ---------------------------------------------------------
    print(f"\n📸 BƯỚC 1: Đọc ảnh từ {image_path}")
    image = Image.open(image_path)
    
    # Processor tự động resize ảnh về 224x224 và chuẩn hóa màu
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs['pixel_values']
    
    print(" 👉 Đầu ra B1 (Tensor điểm ảnh):", pixel_values.shape)
    # Kỳ vọng: [1, 3, 224, 224] -> (1 ảnh, 3 kênh màu RGB, cao 224, rộng 224)

    # ---------------------------------------------------------
    # BƯỚC 2 & 3: Cắt Patch và Trải phẳng (Patching & Flattening)
    # ---------------------------------------------------------
    print("\n🔪 BƯỚC 2 & 3: Mô hình băm ảnh thành lưới và trải phẳng")
    with torch.no_grad():
        # Gọi riêng khối Vision Model (chỉ xử lý ảnh)
        vision_outputs = model.vision_model(pixel_values, output_hidden_states=True)
    
    # Lấy ra trạng thái ở lớp mạng đầu tiên (ngay sau khi cắt patch và gắn mã vị trí)
    initial_embeddings = vision_outputs.hidden_states[0]
    
    print(" 👉 Đầu ra B3 (Chuỗi Patch 1D):", initial_embeddings.shape)
    # Kỳ vọng: [1, 50, 768] (Bản patch32 cắt thành lưới 7x7 = 49 ô + 1 token CLS = 50 toa tàu. Mỗi toa chứa 768 con số đặc trưng).

    # ---------------------------------------------------------
    # BƯỚC 4: Đi qua 12 lớp Mạng nơ-ron (Transformer Encoder)
    # ---------------------------------------------------------
    print("\n🧠 BƯỚC 4: Xử lý Self-Attention qua 12 lớp Transformer")
    # Lấy ra trạng thái sau khi đi qua toàn bộ 12 lớp mạng
    final_hidden_state = vision_outputs.last_hidden_state
    
    print(" 👉 Đầu ra B4 (Ma trận sau 12 lớp):", final_hidden_state.shape)
    # Kích thước vẫn là [1, 50, 768], nhưng các con số bên trong đã chứa ngữ cảnh của toàn bộ bức ảnh.

    # ---------------------------------------------------------
    # BƯỚC 5: Nén lại thành Vector cuối cùng (Vector Embedding)
    # ---------------------------------------------------------
    print("\n🎯 BƯỚC 5: Xuất chuỗi Vector cuối cùng")
    # Trích xuất riêng "toa tàu dẫn đường" (CLS token) ở đầu chuỗi (vị trí số 0)
    # và ném qua một lớp biến đổi (Visual Projection) để ép về chuẩn 512 chiều.
    image_features = model.get_image_features(pixel_values)
    
    print(" 👉 Đầu ra B5 (Image Vector):", image_features.shape)
    # Kỳ vọng: [1, 512] -> Đây chính là Vector để nạp vào Database!
    
    # In thử 5 con số đầu tiên của Vector để xem "hình dáng" của nó
    print(" 🔢 5 giá trị đầu tiên của Vector:", image_features[0][:5].tolist())

if __name__ == "__main__":
    # Thay bằng ảnh của bạn
    debug_clip_vision_steps("anh_o_to.jpg")