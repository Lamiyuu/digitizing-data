import random
import time
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

# --- CẤU HÌNH KẾT NỐI ---
url = "http://localhost:8086"
token = "my-super-secret-auth-token"
org = "my-org"
bucket = "sensor_data"

# Khởi tạo Client
client = InfluxDBClient(url=url, token=token, org=org)
write_api = client.write_api(write_options=SYNCHRONOUS)

print("🚀 Bắt đầu gửi dữ liệu liên tục vào InfluxDB (Nhấn Ctrl+C để dừng)...")

try:
    while True:
        # Mô phỏng dữ liệu ngẫu nhiên
        temp = round(random.uniform(35.0, 42.0), 2)
        cpu = round(random.uniform(20.0, 95.0), 2)

        # Tạo bản ghi (Point) theo chuẩn Line Protocol
        point = (
            Point("system_metrics")
            .tag("host", "server_01")
            .field("temperature", temp)
            .field("cpu_usage", cpu)
        )

        # Ghi vào InfluxDB
        write_api.write(bucket=bucket, org=org, record=point)
        print(f"[SENT] Host: server_01 | Temp: {temp}°C | CPU: {cpu}%")

        # Chờ 2 giây trước khi gửi bản ghi tiếp theo
        time.sleep(2)

except KeyboardInterrupt:
    print("\n🛑 Đã dừng gửi dữ liệu.")
finally:
    client.close()