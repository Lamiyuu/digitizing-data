import os
import cv2
from paddleocr import PaddleOCR

def run_recognition_test():
    # ============================================================
    # BƯỚC 1: THIẾT LẬP ĐƯỜNG DẪN TỰ ĐỘNG THEO MÁY LOCAL
    # ============================================================
    # Tự động lấy đường dẫn gốc của thư mục DIGITIZING DATA
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Trỏ thẳng vào thư mục gốc vì các file inference (json, pdiparams, yml) đang nằm ở đây
    MODEL_DIR = CURRENT_DIR  

    # Trỏ tới file từ điển tiếng Việt vi_dict.txt (bạn nhớ tải và để cùng thư mục này)
    DICT_PATH = os.path.join(CURRENT_DIR, "en_dict.txt") 

    # Thư mục chứa các ảnh dòng chữ cần test
    TEST_IMAGES_DIR = os.path.join(CURRENT_DIR, "img")

    # Tự động tạo thư mục nếu chưa có để tránh lỗi
    if not os.path.exists(TEST_IMAGES_DIR):
        os.makedirs(TEST_IMAGES_DIR)
        print(f"📁 Đã tạo thư mục: {TEST_IMAGES_DIR}")
        print("⚠️ Vui lòng copy vài ảnh chứa một dòng chữ vào thư mục 'test_crops' rồi chạy lại code!")
        return

    # Kiểm tra xem file từ điển đã có chưa
    if not os.path.exists(DICT_PATH):
        print(f"❌ LỖI: Không tìm thấy file {DICT_PATH}")
        print("👉 Hãy tải file vi_dict.txt về và lưu vào thư mục DIGITIZING DATA nhé.")
        return

    # ============================================================
    # BƯỚC 2: KHỞI TẠO AI ENGINE (PP-OCRv5)
    # ============================================================
    print("\n⏳ Đang nạp bộ não AI đọc chữ PP-OCRv5 vào RAM...")
    print("\n⏳ Đang nạp bộ não AI đọc chữ PP-OCRv5 vào RAM...")
    try:
        rec_engine = PaddleOCR(
            device='cpu',  
            text_recognition_model_name="PP-OCRv5_server_rec",  # <--- BÁO CHO AI BIẾT TÊN MODEL V5 CỦA BẠN
            text_recognition_model_dir=MODEL_DIR  
        )
    except Exception as e:
        print(f"❌ Lỗi khởi tạo mô hình: {e}")
        return

    # ============================================================
    # BƯỚC 3: QUÉT THƯ MỤC VÀ ĐỌC CHỮ
    # ============================================================
    print("\n🔍 BẮT ĐẦU ĐỌC CHỮ TỪ THƯ MỤC TEST_CROPS...")
    print("=" * 60)

    # Quét các file ảnh trong thư mục test_crops
    valid_extensions = (".jpg", ".jpeg", ".png")
    test_images = [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith(valid_extensions)]

    if not test_images:
        print("⚠️ Thư mục 'test_crops' đang trống!")
        print("👉 Vui lòng cắt 1-2 dòng chữ từ hóa đơn, lưu thành file ảnh thả vào đó để AI có cái đọc nhé.")
        return

    # Tiến hành đọc từng ảnh
    for img_name in test_images:
        img_path = os.path.join(TEST_IMAGES_DIR, img_name)
        
        # OpenCV đọc ảnh
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Không thể đọc file ảnh: {img_name}")
            continue
            
        # Ném ảnh vào cho AI xử lý
        result = rec_engine.ocr(img, det=False, cls=False)
        
        # Bóc tách kết quả
        if result and result[0]:
            # result[0][0] chứa tuple ('Chữ_đọc_được', Độ_tự_tin)
            text_predicted, confidence = result[0][0]
            print(f"📷 Tên file : {img_name}")
            print(f"   👉 AI Đọc: \"{text_predicted}\"")
            print(f"   📊 Tự tin: {confidence * 100:.2f}%")
        else:
            print(f"📷 Tên file : {img_name}")
            print("   👉 AI Đọc: [KHÔNG NHẬN DIỆN ĐƯỢC CHỮ NÀO]")
            
        print("-" * 60)

    print("✅ HOÀN TẤT KIỂM THỬ!")

if __name__ == "__main__":
    run_recognition_test()