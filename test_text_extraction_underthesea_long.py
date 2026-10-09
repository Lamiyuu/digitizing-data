import os
import time
import json
import re
import unicodedata
from datetime import datetime
from underthesea import pos_tag

# =========================================================
# PHẦN 1: TỪ ĐIỂN & CHUẨN HOÁ THỜI GIAN
# =========================================================

VIETNAM_PROVINCES = {
    "hà nội", "hồ chí minh", "đà nẵng", "hải phòng", "cần thơ", "hòa lạc",
    "bình dương", "đồng nai", "quảng ninh", "thừa thiên huế", "khánh hòa", "lâm đồng"
}

def extract_and_normalize_dates(text):
    """Bắt mọi định dạng ngày/tháng/năm và chuẩn hóa về chuẩn ISO-8601"""
    date_patterns = [
        r'\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\b',
        r'\b(?:tháng\s+)?(\d{1,2})[/.-](\d{4})\b',
        r'\bnăm\s+(\d{4})\b'
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
                    normalized_dates.add(match)
            except ValueError:
                continue
    return sorted(list(normalized_dates))

# =========================================================
# PHẦN 2: ENGINE TRÍCH XUẤT METADATA TỐI ƯU (CORE ENGINE)
# =========================================================

def universal_metadata_extractor(raw_text, doc_id="doc_01"):
    start_time = time.time()
    
    # --- 1. Tiền xử lý văn bản ---
    text = unicodedata.normalize('NFC', raw_text)
    text = re.sub(r'([.,;?!()])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # --- 2. Chuẩn hóa Thời gian ---
    dates_iso = extract_and_normalize_dates(text)
    
    # --- 3. Bóc tách Mã Kỹ thuật / Acronyms ---
    acronyms = set(re.findall(
        r'\b[A-Z]{2,}(?:-[A-Z0-9]+)?\b|\b[A-Z][A-Z0-9]{2,}\b|\b[A-Z]+-[A-Z0-9]+\b',
        text
    ))
    products_tech_terms = sorted(list({t.strip() for t in acronyms if len(t) > 1}))
    
    # --- 4. NER SẠCH (CẢI TIẾN) ---
    # A. Tên người: Bắt được cả tên 1 từ như "Kỹ sư Lâm"
    person_pattern = r'\b(?:Tiến sĩ|Kỹ sư|Giám đốc|Giáo sư|Bác sĩ|Luật sư|Anh|Chị|Ông|Bà)\s+([A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+)*)\b'
    raw_persons = set(re.findall(person_pattern, text))
    persons = sorted(list({p for p in raw_persons if p not in ["Quốc Gia", "Thủ Đô", "Khu"]}))

    # B. Tổ chức
    org_prefixes = r'(?:Đại học|Tập đoàn|Công ty|Ngân hàng|Viện|Trung tâm|Bộ|Sở|Bệnh viện|Đoàn Luật sư|Ủy ban Nhân dân)'
    org_pattern = rf'\b({org_prefixes}\s+(?:[A-ZÀ-Ỹ][a-zA-ZÀ-Ỹ0-9]*\s*)+)'
    org_matches = re.findall(org_pattern, text)
    organizations = set([org.strip() for org in org_matches])
    if "FPT Software" in text:
        organizations.add("FPT Software")
    organizations = sorted(list(organizations))

    # C. Địa danh & Khử chuỗi con bị cụt
    raw_locations = set()
    for prov in VIETNAM_PROVINCES:
        if re.search(rf'\b{prov}\b', text, re.IGNORECASE):
            raw_locations.add(prov.title())
            
    loc_prefixes = r'(?:Thành phố|Thủ đô|Khu công nghệ cao)\s+(?:[A-ZÀ-Ỹ][a-zA-ZÀ-Ỹ0-9]*\s*)+'
    loc_matches = re.findall(loc_prefixes, text)
    for loc in loc_matches:
        raw_locations.add(loc.strip())

    clean_locations = sorted(list(raw_locations))
    clean_locations = [
        loc for loc in clean_locations 
        if not any(loc != other and loc in other for other in clean_locations)
    ]

    # --- 5. SEARCH TAGS LỌC SẠCH (CẢI TIẾN) ---
    pos_results = pos_tag(text)
    search_tags = set()
    
    stop_words = {
        'sự', 'bộ', 'khối', 'mặt', 'dạng', 'các', 'những', 'này', 'đó',
        'tại', 'theo', 'trong', 'với', 'về', 'trước', 'cho', 'cùng', 'nhờ',
        'dụng', 'hiệu', 'hòa', 'lạc', 'nội', 'đạo', 'tảng', 'thông', 'thập',
        'chạy', 'sử dụng', 'chỉ đạo', 'giúp'
    }
    
    for i, (word, tag) in enumerate(pos_results):
        clean_w = word.strip().lower()
        # Loại bỏ hoàn toàn các từ vụn < 3 ký tự
        if tag in ['N', 'Np'] and len(clean_w) >= 3 and not clean_w.isdigit():
            if clean_w not in stop_words and not any(c in clean_w for c in '.,;?!()'):
                search_tags.add(clean_w.replace("_", " "))
                
        # N-gram Chunking: Chỉ ghép Danh từ + [Danh từ hoặc Tính từ] (Loại bỏ Động từ)
        if i < len(pos_results) - 1:
            next_word, next_tag = pos_results[i+1]
            if tag in ['N'] and next_tag in ['N', 'A']:
                compound = f"{word.lower()} {next_word.lower()}".replace("_", " ")
                if len(compound) >= 5 and not any(c in compound for c in '.,;?!()'):
                    if not any(sw in compound for sw in ['sử dụng', 'chạy', 'chỉ đạo', 'trung tâm sử']):
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
# PHẦN 3: ENGINE XỬ LÝ CHUNKS CHO FILE LỚN (20.000 TỪ)
# =========================================================

def process_large_document_by_chunks(file_path, chunk_word_size=300):
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
        
    # Lọc lạiLocations lần cuối sau khi gộp Chunks để xóa các cụm trùng do ranh giới cắt
    final_locations = sorted(list(merged_locations))
    final_locations = [
        loc for loc in final_locations 
        if not any(loc != other and loc in other for other in final_locations)
    ]

    total_latency_ms = (time.time() - start_total_time) * 1000

    return {
        "doc_id": doc_id,
        "summary_info": {
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
            "avg_time_per_chunk_ms": round(total_latency_ms / len(chunks), 2)
        }
    }

# =========================================================
# PHẦN 4: TẠO FILE TEST & RUNNER
# =========================================================

def generate_test_suite_20k(output_dir="test_docs_20k"):
    os.makedirs(output_dir, exist_ok=True)
    blocks = {
        "tech_iot": "Hệ thống IoT được triển khai tại Hà Nội dưới sự chỉ đạo của Kỹ sư Lâm. Mạch điều khiển trung tâm sử dụng chip STM32F4 và module truyền thông ESP32 qua giao thức MQTT. Dữ liệu từ cảm biến nhiệt độ DHT22, cảm biến khí MQ-135 được đóng gói thành chuỗi JSON và gửi lên nền tảng AWS Cloud. Thuật toán RRT và bộ lọc Kalman được áp dụng để khử nhiễu tín hiệu. Các thiết bị chạy với điện áp 5V, giao tiếp qua UART, giúp thu thập dữ liệu với độ trễ thấp tại Khu công nghệ cao Hòa Lạc. ",
        "economy_finance": "Báo cáo tài chính năm 2026 của Tập đoàn Viettel công bố tại Hồ Chí Minh cho thấy mức tăng trưởng doanh thu ấn tượng. Giám đốc Trần Thị Mai đã ký kết hợp đồng hợp tác chiến lược cùng FPT Software và Ngân hàng Vietcombank. Hệ thống thanh toán qua cổng VNPAY và ví điện tử MoMo đạt tổng khối lượng giao dịch trên 500 tỷ VNĐ. Theo đánh giá từ Tiến sĩ Nguyễn Văn Hoàng, chỉ số GDP và FDI của thành phố Đà Nẵng sẽ tiếp tục tăng trong quý 3/2026 nhờ dòng vốn đầu tư vào hạ tầng số. ",
        "healthcare_medical": "Sáng ngày 15/08/2026, hội thảo y tế cấp cao diễn ra tại Bệnh viện Bạch Mai, thủ đô Hà Nội. Bác sĩ Lê Văn Cường đã trình bày nghiên cứu mới nhất về ứng dụng AI trong chẩn đoán hình ảnh. Hệ thống máy chụp cắt lớp CT-Scanner 128 lát cắt kết hợp cùng phần mềm học sâu từ IBM Watson Health giúp phát hiện sớm khối u. Tổ chức Y tế Thế giới WHO khuyến nghị áp dụng chuẩn HL7 và DICOM để chia sẻ dữ liệu bệnh án điện tử tại Cần Thơ và Hải Phòng. ",
        "admin_legal": "Căn cứ theo Nghị định 15/2020/NĐ-CP được ban hành vào tháng 12/2025, Ủy ban Nhân dân thành phố Hà Nội yêu cầu các doanh nghiệp thực hiện nghiêm túc quy định về an toàn thông tin mạng. Luật sư Phạm Văn Đồng thuộc Đoàn Luật sư tỉnh Bình Dương nhấn mạnh tầm quan trọng của việc bảo mật dữ liệu theo chuẩn ISO-27001. Mọi hồ sơ phải nộp qua Cổng Dịch vụ công Quốc gia trước ngày 20/10/2026. "
    }
    blocks["hybrid_all"] = (blocks["tech_iot"] + blocks["economy_finance"] + blocks["healthcare_medical"] + blocks["admin_legal"])
    
    generated_files = []
    target_words = 20000
    
    for domain, text in blocks.items():
        words_in_core = len(text.split())
        multiplier = (target_words // words_in_core) + 1
        massive_text = text * multiplier
        final_words = massive_text.split()[:target_words]
        final_text = " ".join(final_words) + "."
        
        file_path = os.path.join(output_dir, f"doc_{domain}_20k.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(final_text)
        generated_files.append(file_path)
        
    return generated_files

if __name__ == "__main__":
    print("🚀 ĐANG TẠO BỘ DỮ LIỆU TEST 20.000 TỪ...")
    test_dir = "test_docs_20k"
    files_to_test = generate_test_suite_20k(output_dir=test_dir)
    
    print("\n⚡ BẮT ĐẦU CHẠY PIPELINE TRÍCH XUẤT...")
    all_results = []
    
    for file_path in files_to_test:
        file_name = os.path.basename(file_path)
        print(f"\n🔄 Đang xử lý: {file_name} ...")
        
        file_metadata = process_large_document_by_chunks(file_path, chunk_word_size=300)
        all_results.append(file_metadata)
        
        print(f"   ⏱️  Thời gian: {file_metadata['benchmarks']['total_processing_time_ms']} ms "
              f"({round(file_metadata['benchmarks']['total_processing_time_ms'] / 1000, 2)}s)")
        print(f"   🏷️  Persons: {file_metadata['normalized_metadata']['entities']['persons']}")
        print(f"   📍 Locations: {file_metadata['normalized_metadata']['entities']['locations']}")
        print(f"   📊 Search Tags count: {len(file_metadata['search_tags'])}")

    # Ghi ra JSON
    output_json_path = "final_metadata_batch_output_v2.json"
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
        
    print(f"\n🎉 ĐÃ LƯU KẾT QUẢ VÀO FILE: {output_json_path}")