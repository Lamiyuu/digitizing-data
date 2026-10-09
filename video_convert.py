import cv2
import torch
import open_clip
from PIL import Image

def process_video_on_cpu(video_path, query_text):
    device = "cpu"
    # 1. Khởi tạo mô hình EVA-CLIP tương tự như trước
    model, _, preprocess = open_clip.create_model_and_transforms('EVA02-B-16', pretrained='merged2b_s8b_b131k', device=device)
    tokenizer = open_clip.get_tokenizer('EVA02-B-16')
    
    # 2. Đọc Video bằng OpenCV
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) # Lấy số khung hình/giây của video gốc
    frame_count = 0
    sampled_vectors = []
    timestamps = []

    print("⏳ Đang quét và băm video theo mốc thời gian...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Chỉ lấy mẫu 1 khung hình sau mỗi 1 giây (để CPU không bị quá tải)
        if frame_count % int(fps) == 0:
            current_second = frame_count // int(fps)
            
            # Chuyển ảnh từ OpenCV (BGR) sang PIL (RGB) để khớp chuẩn EVA
            color_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(color_converted)
            
            # Tính toán Vector cho khung hình này
            image_input = preprocess(pil_image).unsqueeze(0).to(device)
            with torch.no_grad():
                image_feature = model.encode_image(image_input)
                image_feature /= image_feature.norm(dim=-1, keepdim=True)
                
                sampled_vectors.append(image_feature)
                timestamps.append(current_second)
                
        frame_count += 1
    cap.release()

    # 3. Đối chiếu với câu lệnh chữ viết
    text_input = tokenizer([query_text]).to(device)
    with torch.no_grad():
        text_feature = model.encode_text(text_input)
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
    
    # 4. Tìm kiếm giây phút khớp nhất trong Video
    print(f"\n🔍 Kết quả tìm kiếm cụm từ: '{query_text}'")
    for i, img_vec in enumerate(sampled_vectors):
        # Tính Cosine Similarity thô
        score = (img_vec @ text_feature.T).item()
        print(f"🚩 Phát hiện tại giây thứ {timestamps[i]}: Độ khớp {score * 100:.2f}%")

if __name__ == "__main__":
    process_video_on_cpu("street.mp4", "Road")
    pass