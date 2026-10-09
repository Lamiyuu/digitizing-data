import os
import time
import json
import re
import unicodedata
from datetime import datetime
from collections import Counter

# =========================================================
# 1. KHỞI TẠO SPACY VỚI CƠ CHẾ FALLBACK SAFE
# =========================================================
USE_SPACY = False
try:
    import spacy
    print("⏳ Đang tải mô hình spaCy 'vi_core_news_lg'...")
    nlp = spacy.load("vi_core_news_lg")
    USE_SPACY = True
    print("✅ Đã kích hoạt spaCy Dependency Parser thành công!")
except Exception as e:
    print(f"⚠️ Không thể tải spaCy ({e}). Tự động chuyển sang Underthesea NP Chunking...")
    from underthesea import chunk

# =========================================================
# 2. DICTIONARIES & WHITELISTS CHUẨN
# =========================================================

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

ORGANIZATION_WHITELIST = {
    "Bộ Chính trị", "Bộ Công an", "Bộ Quốc phòng", "Bộ Tài chính", 
    "Bộ Kế hoạch và Đầu tư", "Chính phủ", "Quốc hội", "Ủy ban Nhân dân", 
    "Hội đồng Nhân dân", "VnExpress", "Báo VnExpress", "FPT Software", 
    "Vietcombank", "Viettel", "VNPAY", "ASEAN", "IBM Watson Health",
    "Tập đoàn Viettel", "Đại học Quốc gia", "Bệnh viện Bạch Mai", "WHO"
}

# =========================================================
# 3. CHUẨN HÓA VĂN BẢN VÀ ENTITIES (GIỮ NGUYÊN)
# =========================================================

def extract_and_normalize_timestamps(text):
    normalized_dates = set()
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

    pattern_month_year = r'\b(?:tháng\s+)?(\d{1,2})[/.-](\d{4})\b'
    for month, year in re.findall(pattern_month_year, text, re.IGNORECASE):
        try:
            m, y = int(month), int(year)
            if 1900 <= y <= 2099 and 1 <= m <= 12:
                dt = datetime(y, m, 1)
                normalized_dates.add(dt.strftime("%Y-%m"))
        except ValueError:
            continue

    pattern_year = r'\b(?:năm\s+)?(19\d{2}|20\d{2})\b'
    for year in re.findall(pattern_year, text, re.IGNORECASE):
        normalized_dates.add(year)

    return sorted(list(normalized_dates))

def clean_pdf_web_noise_safe(raw_text):
    text = unicodedata.normalize('NFC', raw_text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\b[1-9]/[1-9]\b', ' ', text)
    text = re.sub(r'\(GMT[+-]\d+\)', ' ', text)
    text = re.sub(r'([.,;?!()])', r' \1 ', text)
    return re.sub(r'\s+', ' ', text).strip()

def extract_clean_entities(raw_text):
    dates = extract_and_normalize_timestamps(raw_text)
    text = clean_pdf_web_noise_safe(raw_text)

    acronyms = set(re.findall(r'\b[A-Z]{2,}(?:-[A-Z0-9]+)?\b|\b[A-Z][A-Z0-9]{2,}\b|\b[A-Z]+-[A-Z0-9]+\b', text))
    stop_acronyms = {"HTML", "HTTP", "HTTPS", "URL", "IMG", "PNG", "JPG"}
    codes = sorted(list({t.strip() for t in acronyms if len(t) > 1 and t.upper() not in stop_acronyms}))

    person_titles = r'(?:Đại tướng|Thượng tướng|Trung tướng|Thiếu tướng|Đại tá|Bộ trưởng|Thứ trưởng|Thủ tướng|Chủ tịch|Bí thư|Tiến sĩ|Giáo sư|Kỹ sư|Giám đốc|Bác sĩ|Luật sư|Anh|Chị|Ông|Bà)'
    person_pattern = rf'\b{person_titles}\s+([A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+)+)\b'
    raw_persons = set(re.findall(person_pattern, text))
    persons = sorted(list({p for p in raw_persons if p not in ["Quốc Gia", "Thủ Đô", "Khu", "Trung Ương"]}))

    organizations = set()
    for org in ORGANIZATION_WHITELIST:
        if re.search(rf'\b{re.escape(org)}\b', text, re.IGNORECASE):
            organizations.add(org)
    clean_orgs = sorted(list(organizations))

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

# =========================================================
# 4. TRÍCH XUẤT SEARCH TAGS BẰNG SPACY DEPENDENCY PARSER
# =========================================================

def extract_search_tags_spacy(clean_text):
    doc = nlp(clean_text)
    tags = []
    
    stop_words = {
        "sự", "bộ", "khối", "mặt", "dạng", "các", "những", "này", "đó",
        "tại", "theo", "trong", "với", "về", "trước", "cho", "cùng", "nhờ",
        "ngày", "tháng", "năm", "thứ", "việc", "sẽ", "đã", "đang", "được",
        "thứ tư", "gmt", "http", "https"
    }

    # Gom cụm Danh từ theo cây cú pháp Dependency Tree của spaCy
    for token in doc:
        if token.pos_ == "NOUN":
            compound = [token.text]
            for child in token.children:
                if child.pos_ in ["NOUN", "PROPN", "ADJ"] and not child.is_punct:
                    if child.text.lower() not in stop_words and len(child.text) >= 2:
                        compound.append(child.text)
            
            full_phrase = " ".join(compound).strip().lower()
            if len(full_phrase) >= 4 and not re.search(r'\d', full_phrase):
                if not any(sw in full_phrase for sw in ['thứ tư', 'gmt', 'http']):
                    tags.append(full_phrase)

    return tags

def extract_search_tags_underthesea(clean_text):
    """Fallback Engine nếu máy bị chặn DLL spaCy"""
    chunk_results = chunk(clean_text)
    tags = []
    stop_words = {"sự", "bộ", "các", "những", "này", "đó", "tại", "theo", "trong", "với", "về", "ngày", "tháng", "năm"}
    
    current = []
    for word, pos, chunk_tag in chunk_results:
        clean_w = word.strip().lower().replace("_", " ")
        if chunk_tag in ["B-NP", "I-NP"]:
            if clean_w not in stop_words and not re.search(r'\d', clean_w):
                current.append(clean_w)
        else:
            if current:
                tags.append(" ".join(current).strip())
                current = []
    if current:
        tags.append(" ".join(current).strip())
    return tags

def rank_and_filter_tags(all_raw_tags, top_n=20):
    """
    Hàm tinh lọc: Đếm tần suất (Frequency) + Giới hạn độ dài từ (2-4 từ)
    Cắt gọn chỉ giữ lại Top N tags quan trọng nhất.
    """
    counts = Counter(all_raw_tags)
    filtered = []
    
    for tag, freq in counts.items():
        words = tag.split()
        # Lọc cụm từ chuẩn: từ 2 đến 4 từ
        if 2 <= len(words) <= 4:
            filtered.append({"tag": tag, "frequency": freq})
            
    # Sắp xếp ưu tiên: Tần suất xuất hiện cao hơn -> Độ dài tốt hơn
    sorted_tags = sorted(filtered, key=lambda x: (x["frequency"], len(x["tag"].split())), reverse=True)
    
    # Trả về danh sách Top N tag tinh gọn
    return [item["tag"] for item in sorted_tags[:top_n]]

# =========================================================
# 5. CHẠY PIPELINE HOÀN CHỈNH
# =========================================================

def process_spacy_pipeline(file_path, chunk_word_size=300):
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
    all_raw_tags = []

    print(f"🔄 Đang xử lý {len(chunks)} Chunks...")
    
    for idx, chunk_text in enumerate(chunks):
        # 1. Trích xuất Entities
        ent = extract_clean_entities(chunk_text)
        merged_timestamps.update(ent["timestamps_iso"])
        merged_acronyms.update(ent["acronyms_and_codes"])
        merged_persons.update(ent["persons"])
        merged_locations.update(ent["locations"])
        merged_orgs.update(ent["organizations"])
        
        # 2. Trích xuất Search Tags (Dùng spaCy hoặc Fallback)
        cleaned_chunk = clean_pdf_web_noise_safe(chunk_text)
        if USE_SPACY:
            chunk_tags = extract_search_tags_spacy(cleaned_chunk)
        else:
            chunk_tags = extract_search_tags_underthesea(cleaned_chunk)
            
        all_raw_tags.extend(chunk_tags)

    # Khử trùng địa danh
    final_locations = sorted(list(merged_locations))
    final_locations = [loc for loc in final_locations if not any(loc != other and loc in other for other in final_locations)]

    # Tinh lọc & Ranking lại toàn bộ Search Tags về Top 20 sắc bén
    final_search_tags = rank_and_filter_tags(all_raw_tags, top_n=20)
    total_latency_ms = (time.time() - start_total_time) * 1000

    return {
        "doc_id": doc_id,
        "summary_info": {
            "file_name": os.path.basename(file_path),
            "total_words": total_words,
            "total_chunks_processed": len(chunks),
            "search_tags_count": len(final_search_tags)
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
        "search_tags": final_search_tags,
        "benchmarks": {
            "total_processing_time_ms": round(total_latency_ms, 2)
        }
    }

# =========================================================
# 6. EXECUTION MAIN
# =========================================================

if __name__ == "__main__":
    target_file = "merged_all_docs_game.txt"
    print(f"\n🚀 CHẠY SPACY RANKED PIPELINE CHO FILE: {target_file}\n")
    
    result = process_spacy_pipeline(target_file, chunk_word_size=300)
    
    if result:
        output_file = "final_spacy_ranked_metadata.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            
        print("=" * 60)
        print("📊 KẾT QUẢ KỲ VỌNG TRONG JSON (TỐI ƯU TOP 20 TAGS):")
        print("=" * 60)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\n🎉 ĐÃ LƯU KẾT QUẢ VÀO FILE: {output_file}")