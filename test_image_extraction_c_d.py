import os
import json
import time
from PIL import Image, ExifTags
import torch

# ==============================================================================
# 1. FIX TRÚNG ĐÍCH LỖI FLASH_ATTN TRÊN WINDOWS
# Trả về danh sách rỗng [] để thư viện không báo lỗi NoneType
# ==============================================================================
import transformers.dynamic_module_utils
transformers.dynamic_module_utils.check_imports = lambda filename: []

# Ẩn cảnh báo Symlink của HuggingFace trên Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from transformers import AutoProcessor, AutoModelForCausalLM

# ==========================================
# 2. CẤU HÌNH VÀ TẢI MODEL FLORENCE-2
# ==========================================
print("🔄 Đang tải mô hình Florence-2-base...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

MODEL_ID = "microsoft/Florence-2-base"

# Khởi tạo Model với eager attention (bỏ qua flash_attn)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=TORCH_DTYPE, 
    trust_remote_code=True,
    attn_implementation="eager"
).to(DEVICE)

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
print(f"✅ Đã tải xong Florence-2 trên thiết bị: {DEVICE.upper()}")


# ==========================================
# 3. HÀM LẤY METADATA KỸ THUẬT (EXIF / CÁCH D)
# ==========================================
def extract_technical_metadata(image_path):
    start_time = time.time()
    
    with Image.open(image_path) as img:
        tech_data = {
            "resolution": f"{img.width}x{img.height}",
            "width": img.width,
            "height": img.height,
            "format": img.format,
            "color_mode": img.mode,
            "file_size_kb": round(os.path.getsize(image_path) / 1024, 2)
        }
        
        exif_info = {}
        info = img._getexif()
        if info:
            for tag, value in info.items():
                decoded_tag = ExifTags.TAGS.get(tag, tag)
                if isinstance(value, (int, float, str)):
                    exif_info[decoded_tag] = value
        
        tech_data["exif_details"] = exif_info
        
    execution_time = (time.time() - start_time) * 1000
    return tech_data, execution_time


# ==========================================
# 4. HÀM AI LẤY CAPTION VÀ TAGS (FLORENCE-2 / CÁCH C)
# ==========================================
def extract_ai_metadata(image_path):
    start_time = time.time()
    image = Image.open(image_path).convert("RGB")
    
    def run_florence_task(task_prompt):
        inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(DEVICE, TORCH_DTYPE)
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(
            generated_text, task=task_prompt, image_size=(image.width, image.height)
        )
        return parsed_answer

    # Bước 1: Lấy Detailed Caption
    caption_result = run_florence_task("<MORE_DETAILED_CAPTION>")
    detailed_caption = caption_result["<MORE_DETAILED_CAPTION>"]

    # Bước 2: Lấy Object Detection (Tags)
    od_result = run_florence_task("<OD>")
    tags = []
    if "<OD>" in od_result and "labels" in od_result["<OD>"]:
        tags = list(set(od_result["<OD>"]["labels"])) # Dùng set để loại bỏ nhãn trùng
        
    execution_time = (time.time() - start_time) * 1000
    return detailed_caption, tags, execution_time


# ==========================================
# 5. KỊCH BẢN MAIN THỰC THI
# ==========================================
def run_test_pipeline(doc_id, image_path):
    print("\n" + "="*50)
    print(f"🚀 BẮT ĐẦU TEST LẤY METADATA CHO TÀI LIỆU ID: {doc_id}")
    print(f"🖼️ Target File: {image_path}")
    print("="*50)
    
    if not os.path.exists(image_path):
        print(f"❌ File {image_path} không tồn tại. Vui lòng kiểm tra lại đường dẫn!")
        return

    # Lấy dữ liệu D
    tech_metadata, d_time = extract_technical_metadata(image_path)
    print(f"⚡ [Technical EXIF] Hoàn thành trong: {d_time:.2f} ms")

    # Lấy dữ liệu C (Caption & Tags)
    caption, tags, c_time = extract_ai_metadata(image_path)
    print(f"🧠 [Florence-2 AI] Hoàn thành (Caption & Tags) trong: {c_time:.2f} ms")

    # Đóng gói Metadata chuẩn bị nạp DB
    unified_metadata = {
        "doc_id": doc_id,
        "file_name": os.path.basename(image_path),
        "file_type": "image",
        
        # Dùng cho Database truy vấn SQL/NoSQL (Postgres/MongoDB)
        "search_tags": tags,
        
        # Dùng cho Vector Database (Qdrant)
        "semantic_context": caption,
        
        # Dữ liệu kỹ thuật lưu kèm
        "technical_metadata": tech_metadata,
        
        "pipeline_benchmarks": {
            "tech_extraction_ms": round(d_time, 2),
            "ai_extraction_ms": round(c_time, 2)
        }
    }

    print("\n📊 KẾT QUẢ JSON METADATA HOÀN CHỈNH:")
    print(json.dumps(unified_metadata, indent=2, ensure_ascii=False))

    return unified_metadata

if __name__ == "__main__":
    TEST_IMAGE = "anh_o_to.jpg" 
    
    # Tạo ảnh mẫu nếu chưa có
    if not os.path.exists(TEST_IMAGE):
        dummy_img = Image.new('RGB', (1280, 720), color = (73, 109, 137))
        dummy_img.save(TEST_IMAGE)
        print(f"⚠️ Đã tạo tạm file ảnh test mẫu: {TEST_IMAGE}. Hãy thay bằng 1 ảnh thật (ví dụ ảnh đường phố) để thấy AI nhận diện chính xác!")

    result_data = run_test_pipeline(doc_id=5001, image_path=TEST_IMAGE)