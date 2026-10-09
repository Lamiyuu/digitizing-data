import os
import time
import json
import re
import unicodedata
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# =========================================================
# PHẦN 1: KHỞI TẠO MODEL QWEN2.5 (1.5B INSTRUCT)
# =========================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

print("⏳ Đang tải mô hình Qwen2.5-1.5B-Instruct...")
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None
)
if device == "cpu":
    model.to("cpu")

print(f"✅ Đã tải xong Model trên thiết bị: {device.upper()}")

# =========================================================
# PHẦN 2: TIỀN XỬ LÝ VĂN BẢN & REGEX HỖ TRỢ
# =========================================================

def clean_pdf_web_noise(raw_text):
    """Tiền xử lý nâng cao: Xóa rác OCR, URL, Header/Footer"""
    text = unicodedata.normalize('NFC', raw_text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', ' ', text)
    text = re.sub(r'\b\d{1,2}:\d{2}\s*(?:AM|PM)?\b', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d{1,2}/\d{1,2}\b', ' ', text)
    text = re.sub(r'\(GMT[+-]\d+\)', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_and_normalize_dates(text):
    """Bắt mọi định dạng ngày/tháng/năm bằng Regex và chuyển về ISO-8601"""
    date_patterns = [
        r'\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\b',
        r'\b(?:tháng\s+)?(\d{1,2})[/.-](\d{4})\b',
        r'\b(?:năm\s+)(\d{4})\b'
    ]
    normalized_dates = set()
    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                if isinstance(match, tuple) and len(match) == 3:
                    dt = datetime.strptime(f"{match[0]}-{match[1]}-{match[2]}", "%d-%m-%Y")
                    normalized_dates.add(dt.strftime("%Y-%m-%d"))
                elif isinstance(match, tuple) and len(match) == 2:
                    dt = datetime.strptime(f"01-{match[0]}-{match[1]}", "%d-%m-%Y")
                    normalized_dates.add(dt.strftime("%Y-%m"))
                elif isinstance(match, str):
                    if 1900 <= int(match) <= 2099:
                        normalized_dates.add(match)
            except ValueError:
                continue
    return sorted(list(normalized_dates))

# =========================================================
# PHẦN 3: QWEN2.5 EXTRACTION ENGINE
# =========================================================

def extract_metadata_with_qwen(text):
    """Sử dụng Qwen2.5-1.5B để bóc tách Thực thể và Tags dưới dạng JSON"""
    
    prompt = f"""Bạn là một chuyên gia bóc tách dữ liệu văn bản tiếng Việt. 
Hãy phân tích đoạn văn bản dưới đây và trích xuất thông tin theo đúng định dạng JSON sau:

{{
  "acronyms_and_codes": ["các mã kỹ thuật, từ viết tắt như FPT, AI, VNPAY..."],
  "persons": ["tên các cá nhân/nhân vật được nhắc đến"],
  "locations": ["tên tỉnh thành, địa danh, quốc gia"],
  "organizations": ["tên công ty, bộ ngành, tập đoàn, trường học"],
  "search_tags": ["5 đến 10 từ khóa/cụm từ quan trọng đại diện cho nội dung văn bản"]
}}

Lưu ý:
- Chỉ trả về duy nhất chuỗi JSON hợp lệ, không viết thêm lời mở đầu hay giải thích.
- Nếu không tìm thấy thông tin cho mục nào, hãy để mảng rỗng [].

Văn bản:
\"\"\"
{text}
\"\"\"
"""

    messages = [
        {"role": "system", "content": "You are a helpful assistant that outputs strictly JSON."},
        {"role": "user", "content": prompt}
    ]
    
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    inputs = tokenizer([formatted_prompt], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,  # Đặt thấp để tăng tính ổn định
            do_sample=False
        )

    response_ids = outputs[0][len(inputs.input_ids[0]):]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

    # Thao tác parse JSON an toàn từ phản hồi của LLM
    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            extracted_json = json.loads(json_match.group(0))
        else:
            extracted_json = json.loads(response_text)
    except Exception:
        # Fallback trong trường hợp model không sinh đúng JSON
        extracted_json = {
            "acronyms_and_codes": [],
            "persons": [],
            "locations": [],
            "organizations": [],
            "search_tags": []
        }

    return extracted_json

def universal_metadata_extractor(raw_text, doc_id="doc_01"):
    start_time = time.time()
    
    # 1. Làm sạch
    text = clean_pdf_web_noise(raw_text)
    
    # 2. Bắt thời gian bằng Regex
    dates_iso = extract_and_normalize_dates(text)
    
    # 3. Trích xuất Metadata bằng Qwen2.5
    llm_metadata = extract_metadata_with_qwen(text)
    
    latency = (time.time() - start_time) * 1000

    return {
        "doc_id": doc_id,
        "normalized_metadata": {
            "timestamps_iso": dates_iso,
            "acronyms_and_codes": sorted(list(set(llm_metadata.get("acronyms_and_codes", [])))),
            "entities": {
                "persons": sorted(list(set(llm_metadata.get("persons", [])))),
                "locations": sorted(list(set(llm_metadata.get("locations", [])))),
                "organizations": sorted(list(set(llm_metadata.get("organizations", []))))
            }
        },
        "search_tags": sorted(list(set(llm_metadata.get("search_tags", [])))),
        "benchmarks": {
            "extraction_time_ms": round(latency, 2)
        }
    }

# =========================================================
# PHẦN 4: PROCESS CHUNKS CHO FILE BẤT KỲ
# =========================================================

def process_single_file(file_path, chunk_word_size=300):
    if not os.path.exists(file_path):
        print(f"❌ Lỗi: Không tìm thấy file '{file_path}'!")
        return None

    start_total_time = time.time()
    doc_id = os.path.basename(file_path).split('.')[0]
    
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
        
    words = raw_text.split()
    total_words = len(words)
    
    chunks = [
        " ".join(words[i:i + chunk_word_size])
        for i in range(0, total_words, chunk_word_size)
    ]
    
    merged_timestamps, merged_acronyms = set(), set()
    merged_persons, merged_locations, merged_orgs = set(), set(), set()
    merged_tags = set()
    
    print(f"⚙️ Đang xử lý {len(chunks)} chunks bằng Qwen2.5...")
    for idx, chunk_text in enumerate(chunks):
        chunk_id = f"{doc_id}_chunk_{idx + 1}"
        print(f"   -> Đang xử lý chunk {idx + 1}/{len(chunks)}...")
        
        meta = universal_metadata_extractor(chunk_text, doc_id=chunk_id)
        
        merged_timestamps.update(meta["normalized_metadata"]["timestamps_iso"])
        merged_acronyms.update(meta["normalized_metadata"]["acronyms_and_codes"])
        merged_persons.update(meta["normalized_metadata"]["entities"]["persons"])
        merged_locations.update(meta["normalized_metadata"]["entities"]["locations"])
        merged_orgs.update(meta["normalized_metadata"]["entities"]["organizations"])
        merged_tags.update(meta["search_tags"])

    total_latency_ms = (time.time() - start_total_time) * 1000

    return {
        "doc_id": doc_id,
        "summary_info": {
            "file_name": os.path.basename(file_path),
            "total_words": total_words,
            "total_chunks_processed": len(chunks),
            "chunk_size_words": chunk_word_size,
            "model_used": MODEL_NAME
        },
        "normalized_metadata": {
            "timestamps_iso": sorted(list(merged_timestamps)),
            "acronyms_and_codes": sorted(list(merged_acronyms)),
            "entities": {
                "persons": sorted(list(merged_persons)),
                "locations": sorted(list(merged_locations)),
                "organizations": sorted(list(merged_orgs))
            }
        },
        "search_tags": sorted(list(merged_tags)),
        "benchmarks": {
            "total_processing_time_ms": round(total_latency_ms, 2),
            "avg_time_per_chunk_ms": round(total_latency_ms / len(chunks), 2) if chunks else 0
        }
    }

# =========================================================
# PHẦN 5: THỰC THI (MAIN)
# =========================================================

if __name__ == "__main__":
    target_file = "merged_all_docs.txt"
    
    print(f"🚀 BẮT ĐẦU PIPELINE QWEN2.5 CHO FILE: {target_file} ...")
    
    result = process_single_file(target_file, chunk_word_size=300)
    
    if result:
        output_json_path = f"output_{result['doc_id']}_qwen.json"
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            
        print("\n" + "=" * 60)
        print("📊 KẾT QUẢ TRÍCH XUẤT TỪ QWEN2.5 (1.5B):")
        print("=" * 60)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\n🎉 Đã lưu kết quả vào file: {output_json_path}")