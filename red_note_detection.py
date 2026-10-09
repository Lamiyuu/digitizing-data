import cv2
import re
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

# 1. Cấu hình mô hình VietOCR (dùng kiến trúc vgg_transformer nhẹ, chuẩn tiếng Việt)
config = Cfg.load_config_from_name('vgg_transformer')
# Nếu có GPU CUDA thì đổi thành 'cuda:0', mặc định dùng 'cpu'
config['device'] = 'cpu'
config['predictor']['beamsearch'] = False

detector = Predictor(config)

def test_first_page_vietocr(image_path: str):
    # 2. Đọc ảnh và cắt 25% phần đầu trang (Header)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Lỗi: Không tìm thấy ảnh tại '{image_path}'")
        return

    h, w, _ = img_bgr.shape
    header_crop = img_bgr[0:int(h * 0.25), 0:w]

    # 3. Chuyển BGR (OpenCV) sang RGB (PIL Image) để đưa vào VietOCR
    header_rgb = cv2.cvtColor(header_crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(header_rgb)

    # 4. Dự đoán văn bản
    extracted_text = detector.predict(pil_img)

    print("=" * 40)
    print("VĂN BẢN TRÍCH XUẤT TỪ HEADER:")
    print("=" * 40)
    print(extracted_text)
    print("=" * 40)

    # 5. Khớp Regex nhận diện trang đầu
    text_clean = extracted_text.upper()
    patterns = {
        "Quốc hiệu": r"CỘNG\s*HÒA\s*XÃ\s*HỘI\s*CHỦ\s*NGHĨA",
        "Tiêu ngữ": r"ĐỘC\s*LẬP\s*-\s*TỰ\s*DO\s*-\s*HẠNH\s*PHÚC",
        "Số hiệu": r"SỐ\s*:\s*[\d\w/]+",
        "Loại văn bản": r"(QUYẾT\s*ĐỊNH|THÔNG\s*BÁO|KẾ\s*HOẠCH|CÔNG\s*VĂN|TỜ\s*TRÌNH)",
        "Trích yếu": r"V/V|VỀ\s*VIỆC"
    }

    matched_keys = [k for k, p in patterns.items() if re.search(p, text_clean)]

    # 6. Kết luận
    print("KẾT QUẢ PHÂN TÍCH:")
    if matched_keys:
        print(f"=> [TRANG ĐẦU] Khớp: {', '.join(matched_keys)}")
    else:
        print("=> [TRANG THƯỜNG / TRANG NỐI TIẾP]")

if __name__ == "__main__":
    TEST_IMAGE = "34319924_88884250797.jpg"
    test_first_page_vietocr(TEST_IMAGE)