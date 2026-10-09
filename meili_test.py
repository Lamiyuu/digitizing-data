import meilisearch

def run_meilisearch_pipeline():
    # Kết nối tới Server bằng chìa khóa ngẫu nhiên từ log của bạn
    MY_MASTER_KEY = "ioKNgUizFRybZTg40U_tEJeQO0u3uXVIoZ_LrdK7nt8"
    client = meilisearch.Client('http://localhost:7700', MY_MASTER_KEY)
    
    # Đặt tên bảng dữ liệu (Index) là 'invoices'
    index_name = 'invoices'
    
    print("🏗️ 1. Khởi tạo Index trong Meilisearch...")
    try:
        client.delete_index(index_name)
    except:
        pass
        
    # Bước 1: Tạo index 
    client.create_index(index_name, {'primaryKey': 'id'})
    # Bước 2: Trỏ biến index 
    index = client.index(index_name)

    # 2. GIAI ĐOẠN INGESTION: Nạp văn bản mẫu
    print("📥 2. Đang nạp dữ liệu hóa đơn mẫu vào hệ thống...")
    documents = [
        {
            "id": "HD12345",
            "title": "Hóa đơn Công ty Thành Đạt",
            "content": "Dịch vụ 1 Mô tả 3.000.000đ. Tổng cộng: 12.000.000đ. Ngân hàng VIB."
        },
        {
            "id": "HD99999",
            "title": "Hóa đơn sửa chữa xe máy",
            "content": "Thay lốp xe máy, bơm hơi, thay nhớt định kỳ. Tổng cộng: 450.000đ."
        }
    ]
    # Nhận về thông tin Task từ hệ thống
    task = index.add_documents(documents)
    print(f"⏳ Đã gửi yêu cầu nạp dữ liệu (Task ID: {task.task_uid}). Đang đợi Meilisearch băm chữ...")
    
    # BẮT BUỘC: Ép Python dừng lại đợi cho đến khi Meilisearch xử lý xong Task này
    client.wait_for_task(task.task_uid)
    print("✅ Đã nạp và lập chỉ mục (Indexing) dữ liệu thành công.")

    # 3. GIAI ĐOẠN SEARCH: Kiểm tra tính năng sửa lỗi chính tả (Typo Tolerance)
    print("\n🔍 3. Thực hiện tìm kiếm thử nghiệm...")
    
    # Cố tình gõ sai chính tả để test: "xê mays"
    search_query = "xê mays" 
    print(f"Từ khóa tìm kiếm: '{search_query}'")
    
    results = index.search(search_query)
    
    print("\n📊 KẾT QUẢ TÌM THẤY TỪ MEILISEARCH:")
    print("-" * 50)
    for hit in results['hits']:
        print(f"📌 ID: {hit['id']} | Tiêu đề: {hit['title']}")
        print(f"   Nội dung: {hit['content']}")
    print("-" * 50)

if __name__ == "__main__":
    run_meilisearch_pipeline()