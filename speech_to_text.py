import os
import time
from faster_whisper import WhisperModel


def run_speech_to_text_test(audio_path):
    # 1. Kiểm tra file cấu phần đầu vào (Ingest)
    if not os.path.exists(audio_path):
        print(f"❌ Không tìm thấy file test tại: {audio_path}")
        print("👉 Vui lòng chuẩn bị 1 file audio và đổi tên thành 'test_voice.mp3'.")
        return

    # 2. Khởi tạo mô hình bản nhỏ tối ưu cho CPU
    model_size = "base"
    print(f"🤖 Đang tải mô hình Whisper [{model_size.upper()}] chạy trên CPU...")
    
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    print("✅ Tải mô hình thành công!")
    print("-" * 60)

    # 3. Tiến hành giải mã âm thanh
    print(f"⏳ Đang xử lý file âm thanh: {audio_path}...")
    start_time = time.time()

    # beam_size=3: Cân bằng giữa tốc độ xử lý của CPU và độ chính xác của từ
    segments, info = model.transcribe(audio_path, beam_size=3)

    print(f"📢 Ngôn ngữ phát hiện trong file: {info.language} (Độ tự tin: {info.language_probability:.2f})")
    print("-" * 60)
    print("📝 KẾT QUẢ ĐẦU RA (TỰ ĐỘNG ĐÁNH CHỈ MỤC THỜI GIAN):")
    
    # 4. In kết quả theo cấu trúc chuỗi thời gian (Time-series Indexing)
    for segment in segments:
        # Định dạng lại thời gian hiển thị dạng Phút:Giây
        start_fmt = f"{int(segment.start // 60):02d}:{int(segment.start % 60):02d}"
        end_fmt = f"{int(segment.end // 60):02d}:{int(segment.end % 60):02d}"
        
        print(f"[{start_fmt} -> {end_fmt}] {segment.text}")

    end_time = time.time()
    print("-" * 60)
    print(f"⚡ Tổng thời gian CPU xử lý: {end_time - start_time:.2f} giây.")

if __name__ == "__main__":
    # Chỉ định file test
    FILE_TEST = "022878.wav"
    run_speech_to_text_test(FILE_TEST)