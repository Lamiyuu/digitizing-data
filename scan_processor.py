import os
import re
from PIL import Image, ImageEnhance
import pytesseract

# ---------------------------------------------------------
# CẤU HÌNH ĐƯỜNG DẪN TESSERACT (Sửa lại nếu máy bạn cài ở ổ khác)
# ---------------------------------------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_raw_text(image_path):
    """
    Bước 1: Đọc ảnh, tăng cường độ tương phản và ép Tesseract 
    đọc theo từng hàng ngang (PSM 6) để giữ cấu trúc.
    """
    print(f"⏳ Đang xử lý ảnh: {image_path}...")
    if not os.path.exists(image_path):
        return None

    # Chuyển ảnh sang đen trắng và tăng tương phản gấp 2.5 lần
    img = Image.open(image_path).convert('L')
    img = ImageEnhance.Contrast(img).enhance(2.5)
    
    # Cấu hình PSM 6 (Đọc theo khối ngang) và ngôn ngữ Việt + Anh
    custom_config = r'--psm 6 -l vie+eng'
    
    # Trích xuất ra văn bản thô
    raw_text = pytesseract.image_to_string(img, config=custom_config)
    return raw_text

def parse_invoice_data(raw_text):
    """
    Bước 2: Sử dụng Regex để quét và bóc tách các trường thông tin 
    có giá trị từ đống văn bản thô.
    """
    data = {
        "invoice_number": None,
        "created_date": None,
        "customer_phone": None,
        "total_amount": None,
        "bank_account": None
    }
    
    # 1. Tìm số hóa đơn (Mẫu: "Hóa đơn số 12345")
    invoice_num_match = re.search(r"Hóa\s*đơn\s*số\s*[:\s]*(\d+)", raw_text, re.IGNORECASE)
    if invoice_num_match:
        data["invoice_number"] = invoice_num_match.group(1)
        
    # 2. Tìm ngày tháng (Mẫu: "Ngày 1 tháng 12 năm 2025")
    date_match = re.search(r"Ngày\s*(\d+)\s*tháng\s*(\d+)\s*năm\s*(\d+)", raw_text, re.IGNORECASE)
    if date_match:
        data["created_date"] = f"{date_match.group(1)}/{date_match.group(2)}/{date_match.group(3)}"
        
    # 3. Tìm số điện thoại (+84 912 345 678 hoặc 0912...)
    phone_match = re.search(r"(?:\+84|0)\s*\d{2,4}\s*\d{3}\s*\d{3,4}", raw_text)
    if phone_match:
        # Xóa khoảng trắng thừa trong số điện thoại
        data["customer_phone"] = phone_match.group(0).replace(" ", "")

    # 4. Tìm tổng tiền (Mẫu: "TỔNG CỘNG: 12.000.000Đ")
    total_match = re.search(r"TỔNG\s*CỘNG\s*[:\s]*([\d\.,]+)", raw_text, re.IGNORECASE)
    if total_match:
        data["total_amount"] = total_match.group(1)
        
    # 5. Tìm số tài khoản ngân hàng (Mẫu: "Số tài khoản: 123-456-7890")
    stk_match = re.search(r"Số\s*tài\s*khoản\s*[:\s]*([\d-]+)", raw_text, re.IGNORECASE)
    if stk_match:
        data["bank_account"] = stk_match.group(1)

    return data

def process_single_invoice(image_path):
    """Hàm chạy luồng xử lý chính"""
    print("-" * 50)
    # Bước 1: Lấy Text
    raw_text = extract_raw_text(image_path)
    
    if not raw_text:
        print("❌ Lỗi: Không thể đọc được file ảnh.")
        return
        
    # Bước 2: Bóc tách Data
    parsed_data = parse_invoice_data(raw_text)
    
    # Bước 3: Hiển thị kết quả chuẩn JSON/Dictionary
    print("\n✅ KẾT QUẢ BÓC TÁCH DỮ LIỆU:")
    print("-" * 50)
    for key, value in parsed_data.items():
        # Định dạng in ra cho đẹp mắt
        print(f"{key:<18}: {value if value else 'Không tìm thấy'}")
    print("-" * 50)

if __name__ == "__main__":
    # Đảm bảo file ảnh image_e9b39f.jpg nằm cùng thư mục với file code này
    test_image_path = "1131w-Ks9pl47qmWo.webp"
    process_single_invoice(test_image_path)