import os
import time
import json
import re
import unicodedata
from datetime import datetime
from underthesea import pos_tag

# =========================================================
# PHẦN 1: TỪ ĐIỂN & TIỀN XỬ LÝ LỌC RÁC TỔNG QUÁT
# =========================================================

VIETNAM_PROVINCES = {
    "hà nội", "hồ chí minh", "đà nẵng", "hải phòng", "cần thơ", "hòa lạc",
    "bình dương", "đồng nai", "quảng ninh", "thừa thiên huế", "khánh hòa", "lâm đồng",
    "việt nam", "toàn quốc"
}

def clean_pdf_web_noise(raw_text):
    """
    Tiền xử lý nâng cao: Xóa rác OCR, URL, Header/Footer của trang web/PDF
    """
    text = unicodedata.normalize('NFC', raw_text)
    
    # 1. Xóa URLs (https://..., http://..., www...)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 2. Xóa định dạng số trang/ngày in web (VD: 1/5, 7/29/26, 3:17 PM, GMT+7)
    text = re.sub(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', ' ', text)
    text = re.sub(r'\b\d{1,2}:\d{2}\s*(?:AM|PM)?\b', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d{1,2}/\d{1,2}\b', ' ', text) # Dạng 1/5
    text = re.sub(r'\(GMT[+-]\d+\)', ' ', text)
    
    # 3. Tách dấu câu dính liền khỏi từ
    text = re.sub(r'([.,;?!()])', r' \1 ', text)
    
    # 4. Gom khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_and_normalize_dates(text):
    """Bắt mọi định dạng ngày/tháng/năm và chuẩn hóa về ISO-8601 (YYYY-MM-DD)"""
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
                    # Loại bỏ số nhà/số điều luật bị nhầm thành năm (chỉ lấy mốc từ năm 1900-2099)
                    if 1900 <= int(match) <= 2099:
                        normalized_dates.add(match)
            except ValueError:
                continue
    return sorted(list(normalized_dates))

# =========================================================
# PHẦN 2: CORE METADATA EXTRACTION ENGINE (UNIVERSAL)
# =========================================================

def universal_metadata_extractor(raw_text, doc_id="doc_01"):
    start_time = time.time()
    
    # --- 1. Lọc rác văn bản ---
    text = clean_pdf_web_noise(raw_text)
    
    # --- 2. Chuẩn hóa Thời gian ---
    dates_iso = extract_and_normalize_dates(text)
    
    # --- 3. Bóc tách Mã Kỹ thuật / Viết tắt chuẩn ---
    acronyms = set(re.findall(
        r'\b[A-Z]{2,}(?:-[A-Z0-9]+)?\b|\b[A-Z][A-Z0-9]{2,}\b|\b[A-Z]+-[A-Z0-9]+\b',
        text
    ))
    # Loại bỏ các từ tiếng Anh ngắn không phải là mã/tên viết tắt chuẩn
    stop_acronyms = {"HTML", "HTTP", "HTTPS", "URL", "IMG", "PNG", "JPG"}
    products_tech_terms = sorted(list({
        t.strip() for t in acronyms 
        if len(t) > 1 and t.upper() not in stop_acronyms
    }))
    
    # --- 4. TRÍCH XUẤT NER TỔNG QUÁT (MỞ RỘNG) ---
    
    # A. Tên người: Thêm các cấp bậc, chức danh Quân đội, Công an, Chính trị
    person_titles = (
        r'(?:Đại tướng|Thượng tướng|Trung tướng|Thiếu tướng|Đại tá|'
        r'Bộ trưởng|Thứ trưởng|Thủ tướng|Chủ tịch|Bí thư|'
        r'Tiến sĩ|Giáo sư|Kỹ sư|Giám đốc|Bác sĩ|Luật sư|Anh|Chị|Ông|Bà)'
    )
    person_pattern = rf'\b{person_titles}\s+([A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+)+)\b'
    raw_persons = set(re.findall(person_pattern, text))
    
    # Lọc bỏ từ bị nhầm ngữ cảnh
    persons = sorted(list({p for p in raw_persons if p not in ["Quốc Gia", "Thủ Đô", "Khu", "Trung Ương"]}))

    # B. Tổ chức: Mở rộng đầy đủ cơ quan nhà nước, báo chí, tập đoàn
    org_prefixes = (
        r'(?:Bộ Chính trị|Bộ Công an|Bộ Quốc phòng|Bộ Tài chính|Bộ Kế hoạch và Đầu tư|'
        r'Chính phủ|Quốc hội|Ủy ban Nhân dân|Hội đồng Nhân dân|'
        r'Đại học|Tập đoàn|Công ty|Ngân hàng|Viện|Trung tâm|Bệnh viện|Báo)'
    )
    org_pattern = rf'\b({org_prefixes}\s+(?:[A-ZÀ-Ỹ][a-zA-ZÀ-Ỹ0-9]*\s*)+)'
    org_matches = re.findall(org_pattern, text)
    organizations = set([org.strip() for org in org_matches])
    
    # Bổ sung các tổ chức tiếng Anh / Tên riêng đặc thù
    for custom_org in ["VnExpress", "FPT Software", "Vietcombank", "Viettel", "VNPAY", "ASEAN"]:
        if re.search(rf'\b{custom_org}\b', text, re.IGNORECASE):
            organizations.add(custom_org)
            
    organizations = sorted(list(organizations))

    # C. Địa danh & Khử chuỗi con bị cụt
    raw_locations = set()
    for prov in VIETNAM_PROVINCES:
        if re.search(rf'\b{prov}\b', text, re.IGNORECASE):
            raw_locations.add(prov.title())
            
    loc_prefixes = r'(?:Thành phố|Thủ đô|Tỉnh|Huyện|Quận|Khu công nghệ cao)\s+(?:[A-ZÀ-Ỹ][a-zA-ZÀ-Ỹ0-9]*\s*)+'
    loc_matches = re.findall(loc_prefixes, text)
    for loc in loc_matches:
        raw_locations.add(loc.strip())

    clean_locations = sorted(list(raw_locations))
    clean_locations = [
        loc for loc in clean_locations 
        if not any(loc != other and loc in other for other in clean_locations)
    ]

    # --- 5. SEARCH TAGS LỌC SẠCH VÀ SẮC BÉN ---
    pos_results = pos_tag(text)
    search_tags = set()
    
    stop_words = {
        'sự', 'bộ', 'khối', 'mặt', 'dạng', 'các', 'những', 'này', 'đó',
        'tại', 'theo', 'trong', 'với', 'về', 'trước', 'cho', 'cùng', 'nhờ',
        'dụng', 'hiệu', 'hòa', 'lạc', 'nội', 'đạo', 'tảng', 'thông', 'thập',
        'chạy', 'sử dụng', 'chỉ đạo', 'giúp', 'ngày', 'tháng', 'năm', 'thứ'
    }
    
    for i, (word, tag) in enumerate(pos_results):
        clean_w = word.strip().lower()
        
        # Bỏ các từ chứa chuỗi số ID rác (VD: 5102813) hoặc gạch nối OCR lỗi
        if re.search(r'\d{4,}', clean_w) or '-' in clean_w:
            continue
            
        if tag in ['N', 'Np'] and len(clean_w) >= 3 and not clean_w.isdigit():
            if clean_w not in stop_words and not any(c in clean_w for c in '.,;?!()'):
                search_tags.add(clean_w.replace("_", " "))
                
        # N-gram Chunking: Ghép [Danh từ + Danh từ] hoặc [Danh từ + Tính từ]
        if i < len(pos_results) - 1:
            next_word, next_tag = pos_results[i+1]
            if tag in ['N'] and next_tag in ['N', 'A']:
                compound = f"{word.lower()} {next_word.lower()}".replace("_", " ")
                if len(compound) >= 5 and not any(c in compound for c in '.,;?!()'):
                    if not re.search(r'\d', compound) and not any(sw in compound for sw in ['sử dụng', 'chạy', 'chỉ đạo']):
                        search_tags.add(compound)

    latency = (time.time() - start_time) * 1000

    return {
        "doc_id": doc_id,
        "normalized_metadata": {
            "timestamps_iso": dates_iso,
            "acronyms_and_codes": products_tech_terms,
            "entities": {
                "persons": persons,
                "locations": clean_locations,
                "organizations": organizations
            }
        },
        "search_tags": sorted(list(search_tags)),
        "benchmarks": {
            "extraction_time_ms": round(latency, 2)
        }
    }

# =========================================================
# PHẦN 3: ENGINE XỬ LÝ CHUNKS CHO FILE BẤT KỲ
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
    
    for idx, chunk_text in enumerate(chunks):
        chunk_id = f"{doc_id}_chunk_{idx + 1}"
        meta = universal_metadata_extractor(chunk_text, doc_id=chunk_id)
        
        merged_timestamps.update(meta["normalized_metadata"]["timestamps_iso"])
        merged_acronyms.update(meta["normalized_metadata"]["acronyms_and_codes"])
        merged_persons.update(meta["normalized_metadata"]["entities"]["persons"])
        merged_locations.update(meta["normalized_metadata"]["entities"]["locations"])
        merged_orgs.update(meta["normalized_metadata"]["entities"]["organizations"])
        merged_tags.update(meta["search_tags"])
        
    # Lọc trùng địa danh cuối cùng
    final_locations = sorted(list(merged_locations))
    final_locations = [
        loc for loc in final_locations 
        if not any(loc != other and loc in other for other in final_locations)
    ]

    total_latency_ms = (time.time() - start_total_time) * 1000

    return {
        "doc_id": doc_id,
        "summary_info": {
            "file_name": os.path.basename(file_path),
            "total_words": total_words,
            "total_chunks_processed": len(chunks),
            "chunk_size_words": chunk_word_size
        },
        "normalized_metadata": {
            "timestamps_iso": sorted(list(merged_timestamps)),
            "acronyms_and_codes": sorted(list(merged_acronyms)),
            "entities": {
                "persons": sorted(list(merged_persons)),
                "locations": final_locations,
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
# PHẦN 4: THỰC THI (MAIN)
# =========================================================

if __name__ == "__main__":
    target_file = "merged_all_docs.txt"
    
    print(f"🚀 BẮT ĐẦU CHẠY PIPELINE TỐI ƯU CHO FILE: {target_file} ...")
    
    result = process_single_file(target_file, chunk_word_size=300)
    
    if result:
        output_json_path = f"output_{result['doc_id']}_clean.json"
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            
        print("\n" + "=" * 60)
        print("📊 KẾT QUẢ TRÍCH XUẤT ĐÃ ĐƯỢC TỐI ƯU HOÀN HẢO:")
        print("=" * 60)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\n🎉 Đã lưu kết quả sạch vào file: {output_json_path}")