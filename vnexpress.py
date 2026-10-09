import glob

# 1. Tìm tất cả các file .txt trong thư mục
txt_files = glob.glob("read/*.txt")

output_file = "merged_all_docs_game.txt"

# 2. Đọc và nối nội dung các file lại với nhau
with open(output_file, "w", encoding="utf-8") as outfile:
    for fpath in txt_files:
        with open(fpath, "r", encoding="utf-8") as infile:
            outfile.write(infile.read() + "\n\n")  # Thêm dòng trống giữa các file

print(f"✅ Đã nối {len(txt_files)} file thành công vào: {output_file}")