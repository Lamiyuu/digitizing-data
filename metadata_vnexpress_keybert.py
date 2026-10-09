import os
import time
import json
import re
import unicodedata
from datetime import datetime
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from underthesea import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

# =========================================================
# 1. KHỞI TẠO KEYBERT VÀ DICTIONARIES CHUẨN
# =========================================================
print("⏳ Đang khởi tạo KeyBERT Model & Underthesea Tokenizer...")
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
kw_model = KeyBERT(model=embedding_model)

# Danh sách chuẩn 34 tỉnh/thành phố trực thuộc Trung ương của Việt Nam sau sáp nhập
VIETNAM_34_PROVINCES = {
    "hà nội", "hồ chí minh", "hải phòng", "đà nẵng", "cần thơ", "hòa lạc",
    "quảng ninh", "bắc ninh", "hải dương", "hưng yên", "thái bình", "nam định",
    "ninh bình", "thanh hóa", "nghệ an", "hà tĩnh", "quảng bình", "quảng trị",
    "thừa thiên huế", "quảng nam", "quảng ngãi", "bình định", "phú yên", "khánh hòa",
    "ninh thuận", "bình thuận", "kon tum", "gia lai", "đắc lắc", "đắk lắk", "đắk nông",
    "lâm đồng", "bình dương", "đồng nai", "bà rịa vũng tàu", "long an", "tiền giang",
    "bến tre", "trà vinh", "vĩnh long", "đồng tháp", "an giang", "kiên giang",
    "hậu giang", "sóc trăng", "bạc lưu", "cà mau", "việt nam", "toàn quốc"
}

# Whitelist các Tổ chức chuẩn xác (Ngăn chặn triệt để rác chữ dính sau)
ORGANIZATION_WHITELIST = {
    "Bộ Chính trị", "Bộ Công an", "Bộ Quốc phòng", "Bộ Tài chính", 
    "Bộ Kế hoạch và Đầu tư", "Chính phủ", "Quốc hội", "Ủy ban Nhân dân", 
    "Hội đồng Nhân dân", "VnExpress", "Báo VnExpress", "FPT Software", 
    "Vietcombank", "Viettel", "VNPAY", "ASEAN", "IBM Watson Health",
    "Tập đoàn Viettel", "Đại học Quốc gia", "Bệnh viện Bạch Mai", "WHO"
}

def vietnamese_tokenizer(text):
    text = unicodedata.normalize('NFC', text)
    tokens = word_tokenize(text, format="text").split()
    return tokens

vietnamese_vectorizer = CountVectorizer(
    tokenizer=vietnamese_tokenizer,
    ngram_range=(1, 3),
    stop_words=[
        'sự', 'bộ', 'các', 'những', 'ngày', 'tháng', 'năm', 'được', 'cho', 
        'với', 'trong', 'tại', 'theo', 'trước', 'sau', 'khi', 'về', 'cùng', 'nhờ'
    ]
)

# =========================================================
# 2. HÀM CHUẨN HÓA THỜI GIAN & LỌC RÁC SAFE
# =========================================================

def extract_and_normalize_timestamps(text):
    """
    Bắt trọn vẹn Ngày/Tháng/Năm và Giờ:Phút từ văn bản gốc (VD: 29/7/2026, 11:17)
    Quy về chuẩn ISO-8601 (YYYY-MM-DDTHH:MM:SS hoặc YYYY-MM-DD)
    """
    normalized_dates = set()
    
    # Pattern 1: Bắt Ngày/Tháng/Năm kèm Giờ:Phút
    pattern_full_time = r'(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})(?:[,\s]+(\d{1,2}):(\d{2}))?'
    matches_full = re.findall(pattern_full_time, text)
    
    for day, month, year, hour, minute in matches_full:
        try:
            d, m, y = int(day), int(month), int(year)
            h = int(hour) if hour else 0
            mi = int(minute) if minute else 0
            
            if 1900 <= y <= 2099 and 1 <= m <= 12 and 1 <= d <= 31:
                dt = datetime(y, m, d, h, mi)
                if hour:
                    normalized_dates.add(dt.strftime("%Y-%m-%dT%H:%M:%S"))
                else:
                    normalized_dates.add(dt.strftime("%Y-%m-%d"))
        except ValueError:
            continue

    # Pattern 2: Bắt "tháng MM/YYYY" hoặc "MM/YYYY"
    pattern_month_year = r'\b(?:tháng\s+)?(\d{1,2})[/.-](\d{4})\b'
    for month, year in re.findall(pattern_month_year, text, re.IGNORECASE):
        try:
            m, y = int(month), int(year)
            if 1900 <= y <= 2099 and 1 <= m <= 12:
                dt = datetime(y, m, 1)
                normalized_dates.add(dt.strftime("%Y-%m"))
        except ValueError:
            continue

    # Pattern 3: Bắt năm độc lập
    pattern_year = r'\b(?:năm\s+)?(19\d{2}|20\d{2})\b'
    for year in re.findall(pattern_year, text, re.IGNORECASE):
        normalized_dates.add(year)

    return sorted(list(normalized_dates))

def clean_pdf_web_noise_safe(raw_text):
    """Lọc rác Web/PDF an toàn, KHÔNG xóa mất ngày tháng của văn bản gốc"""
    text = unicodedata.normalize('NFC', raw_text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\b[1-9]/[1-9]\b', ' ', text) # Chỉ xóa số trang dạng 1/5
    text = re.sub(r'\(GMT[+-]\d+\)', ' ', text)
    text = re.sub(r'([.,;?!()])', r' \1 ', text)
    return re.sub(r'\s+', ' ', text).strip()

# =========================================================
# 3. CORE METADATA EXTRACTION ENGINE (UNIVERSAL)
# =========================================================

def extract_clean_entities(raw_text):
    # 1. Bắt Timestamps trực tiếp trên văn bản gốc
    dates = extract_and_normalize_timestamps(raw_text)
    
    # 2. Làm sạch văn bản cho các bước bóc tách tiếp theo
    text = clean_pdf_web_noise_safe(raw_text)

    # 3. Acronyms
    acronyms = set(re.findall(r'\b[A-Z]{2,}(?:-[A-Z0-9]+)?\b|\b[A-Z][A-Z0-9]{2,}\b|\b[A-Z]+-[A-Z0-9]+\b', text))
    stop_acronyms = {"HTML", "HTTP", "HTTPS", "URL", "IMG", "PNG", "JPG"}
    codes = sorted(list({t.strip() for t in acronyms if len(t) > 1 and t.upper() not in stop_acronyms}))

    # 4. Persons
    person_titles = r'(?:Đại tướng|Thượng tướng|Trung tướng|Thiếu tướng|Đại tá|Bộ trưởng|Thứ trưởng|Thủ tướng|Chủ tịch|Bí thư|Tiến sĩ|Giáo sư|Kỹ sư|Giám đốc|Bác sĩ|Luật sư|Anh|Chị|Ông|Bà)'
    person_pattern = rf'\b{person_titles}\s+([A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+)+)\b'
    raw_persons = set(re.findall(person_pattern, text))
    persons = sorted(list({p for p in raw_persons if p not in ["Quốc Gia", "Thủ Đô", "Khu", "Trung Ương"]}))

    # 5. Organizations (Dùng Whitelist sạch 100%, chống dính rác chữ)
    organizations = set()
    for org in ORGANIZATION_WHITELIST:
        if re.search(rf'\b{re.escape(org)}\b', text, re.IGNORECASE):
            organizations.add(org)
    clean_orgs = sorted(list(organizations))

    # 6. Locations (34 Tỉnh thành + Tiền tố Xã/Huyện/Phường/Thị xã...)
    raw_locs = set()
    for prov in VIETNAM_34_PROVINCES:
        if re.search(rf'\b{prov}\b', text, re.IGNORECASE):
            raw_locs.add(prov.title())
            
    loc_prefixes = r'(?:Thành phố|Thủ đô|Tỉnh|Huyện|Quận|Thị xã|Thị trấn|Phường|Xã|Thôn|Bản|Khu công nghệ cao)\s+(?:[A-ZÀ-Ỹ][a-zA-ZÀ-Ỹ0-9]*\s*)+'
    loc_matches = re.findall(loc_prefixes, text)
    for loc in loc_matches:
        raw_locs.add(loc.strip())
    
    clean_locs = sorted(list(raw_locs))
    clean_locs = [loc for loc in clean_locs if not any(loc != other and loc in other for other in clean_locs)]

    return {
        "timestamps_iso": dates,
        "acronyms_and_codes": codes,
        "persons": persons,
        "locations": clean_locs,
        "organizations": clean_orgs
    }

def extract_keybert_tags(clean_text, top_n=10):
    keywords = kw_model.extract_keywords(
        clean_text,
        vectorizer=vietnamese_vectorizer,
        use_mmr=True,
        diversity=0.5,
        top_n=top_n
    )
    tags_dict = {}
    for kw, score in keywords:
        tag_clean = kw.replace("_", " ").strip().lower()
        if len(tag_clean) > 2 and not tag_clean.isdigit() and not re.search(r'\d{4,}', tag_clean):
            tags_dict[tag_clean] = round(float(score), 4)
    return tags_dict

# =========================================================
# 4. CHẠY PIPELINE TRÊN FILE THEO CHUNKS
# =========================================================

def process_hybrid_pipeline(file_path, chunk_word_size=300):
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
    all_keybert_scores = {}

    print(f"🔄 Đang chạy Pipeline trên {len(chunks)} Chunks...")
    
    for idx, chunk_text in enumerate(chunks):
        # 1. Trích xuất Entities
        ent = extract_clean_entities(chunk_text)
        merged_timestamps.update(ent["timestamps_iso"])
        merged_acronyms.update(ent["acronyms_and_codes"])
        merged_persons.update(ent["persons"])
        merged_locations.update(ent["locations"])
        merged_orgs.update(ent["organizations"])
        
        # 2. Trích xuất Search Tags KeyBERT
        cleaned_chunk = clean_pdf_web_noise_safe(chunk_text)
        chunk_tags = extract_keybert_tags(cleaned_chunk, top_n=8)
        for tag, score in chunk_tags.items():
            if tag not in all_keybert_scores or score > all_keybert_scores[tag]:
                all_keybert_scores[tag] = score

    # Khử trùng chuỗi con ở Địa danh
    final_locations = sorted(list(merged_locations))
    final_locations = [loc for loc in final_locations if not any(loc != other and loc in other for other in final_locations)]

    sorted_search_tags = sorted(all_keybert_scores.items(), key=lambda x: x[1], reverse=True)
    total_latency_ms = (time.time() - start_total_time) * 1000

    return {
        "doc_id": doc_id,
        "summary_info": {
            "file_name": os.path.basename(file_path),
            "total_words": total_words,
            "total_chunks_processed": len(chunks)
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
        "search_tags_ranked": [
            {"tag": tag, "relevance_score": score} for tag, score in sorted_search_tags
        ],
        "benchmarks": {
            "total_processing_time_ms": round(total_latency_ms, 2)
        }
    }

# =========================================================
# 5. EXECUTION MAIN
# =========================================================

if __name__ == "__main__":
    target_file = "merged_all_docs.txt"
    print(f"\n🚀 CHẠY PIPELINE HOÀN CHỈNH CHO FILE: {target_file}\n")
    
    result = process_hybrid_pipeline(target_file, chunk_word_size=300)
    
    if result:
        output_file = "final_clean_metadata.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            
        print("=" * 60)
        print("📊 KẾT QUẢ KỲ VỌNG TRONG JSON (ĐÃ SẠCH BONG CÁC LỖI):")
        print("=" * 60)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\n🎉 ĐÃ LƯU KẾT QUẢ VÀO FILE: {output_file}")