import time
import random
from datetime import datetime, timedelta

import psycopg2
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from neo4j import GraphDatabase
from clickhouse_driver import Client as ClickHouseClient
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

VEHICLE_ID = "TRUCK_101"

# Kết nối
pg_conn = psycopg2.connect(host="localhost", port=5432, user="admin", password="password123", dbname="core_db")
pg_cursor = pg_conn.cursor()

influx_client = InfluxDBClient(url="http://localhost:8086", token="my-super-secret-auth-token", org="my-org")
influx_write_api = influx_client.write_api(write_options=SYNCHRONOUS)

neo4j_driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password123"))

# ClickHouse: Kết nối với user admin
ch_client = ClickHouseClient(host='localhost', port=9000, user='admin', password='password123')
qdrant_client = QdrantClient(url="http://localhost:6333")

def init_databases():
    print("⚙️ Đang khởi tạo bảng và dữ liệu mẫu...", flush=True)
    
    # 1. Postgres
    pg_cursor.execute("""
        CREATE TABLE IF NOT EXISTS vehicles (
            vehicle_id VARCHAR(50) PRIMARY KEY,
            license_plate VARCHAR(20),
            driver_name VARCHAR(100)
        );
        INSERT INTO vehicles (vehicle_id, license_plate, driver_name)
        VALUES (%s, '29C-12345', 'Nguyen Van A')
        ON CONFLICT (vehicle_id) DO NOTHING;
    """, (VEHICLE_ID,))
    pg_conn.commit()

    # 2. Neo4j
    with neo4j_driver.session() as session:
        session.run("""
            MERGE (v:Vehicle {id: $id})
            MERGE (depot:Depot {name: 'Kho Ha Noi'})
            MERGE (v)-[:ASSIGNED_TO]->(depot)
        """, id=VEHICLE_ID)

    # 3. ClickHouse: Bảng được tối ưu dạng Columnar (MergeTree)
    ch_client.execute('''
        CREATE TABLE IF NOT EXISTS vehicle_analytics (
            vehicle_id String,
            timestamp DateTime,
            speed Float32,
            engine_temp Float32
        ) ENGINE = MergeTree()
        ORDER BY (vehicle_id, timestamp)
    ''')

    # --- ĐẶC BIỆT: BƠM SẴN 100,000 BẢN GHI GIẢ LẬP VÀO CLICKHOUSE ---
    current_count = ch_client.execute('SELECT count() FROM vehicle_analytics')[0][0]
    if current_count < 100000:
        print("⚡ Đang bơm nhanh 100,000 bản ghi lịch sử vào ClickHouse...", flush=True)
        bulk_data = []
        base_time = datetime.now() - timedelta(days=30)
        for i in range(100000):
            record_time = base_time + timedelta(seconds=i*20)
            spd = round(random.uniform(30.0, 110.0), 2)
            tmp = round(random.uniform(70.0, 110.0), 2)
            bulk_data.append((VEHICLE_ID, record_time, spd, tmp))
        
        ch_client.execute('INSERT INTO vehicle_analytics (vehicle_id, timestamp, speed, engine_temp) VALUES', bulk_data)
        print(f"✅ Đã bơm xong! Tổng số bản ghi hiện tại trong ClickHouse: {ch_client.execute('SELECT count() FROM vehicle_analytics')[0][0]}", flush=True)

    # 4. Qdrant
    if not qdrant_client.collection_exists("vehicle_logs"):
        qdrant_client.create_collection(
            collection_name="vehicle_logs",
            vectors_config=VectorParams(size=4, distance=Distance.COSINE)
        )
    print("✅ Khởi tạo hệ thống thành công!\n", flush=True)

def run_analytics_query():
    """Hàm chạy câu lệnh OLAP phân tích tốc độ xử lý của ClickHouse"""
    start_time = time.time()
    
    # Lệnh SQL phân tích: Tính tốc độ trung bình, tốc độ cao nhất, và số lần quá nhiệt (>100°C)
    query = """
        SELECT 
            vehicle_id,
            count() as total_records,
            avg(speed) as avg_speed,
            max(speed) as max_speed,
            countIf(engine_temp > 100) as overheat_warnings
        FROM vehicle_analytics
        GROUP BY vehicle_id
    """
    result = ch_client.execute(query)
    execution_time = (time.time() - start_time) * 1000 # tính theo miligiây
    
    print("\n--- 📊 BÁO CÁO PHÂN TÍCH TỪ CLICKHOUSE (OLAP) ---", flush=True)
    for row in result:
        print(f"• Mã xe: {row[0]}")
        print(f"• Tổng số bản ghi đã quét: {row[1]:,} dòng")
        print(f"• Tốc độ trung bình: {row[2]:.2f} km/h")
        print(f"• Vận tốc cực đại: {row[3]:.2f} km/h")
        print(f"• Số lần cảnh báo quá nhiệt (>100°C): {row[4]} lần")
    print(f"⚡ Thời gian ClickHouse xử lý: {execution_time:.2f} ms")
    print("--------------------------------------------------\n", flush=True)

def run_simulation():
    init_databases()
    # Chạy thử 1 câu truy vấn OLAP ngay khi khởi chạy
    run_analytics_query()
    
    print(f"🚀 Bắt đầu luồng ghi thời gian thực cho {VEHICLE_ID} (Nhấn Ctrl+C để dừng)...\n", flush=True)
    try:
        step = 0
        while True:
            current_time = datetime.now()
            speed = round(random.uniform(40.0, 90.0), 2)
            temp = round(random.uniform(75.0, 105.0), 2)

            # InfluxDB
            point = Point("telemetry").tag("vehicle_id", VEHICLE_ID).field("speed", speed).field("engine_temp", temp)
            influx_write_api.write(bucket="sensor_data", org="my-org", record=point)

            # ClickHouse
            ch_client.execute(
                'INSERT INTO vehicle_analytics (vehicle_id, timestamp, speed, engine_temp) VALUES',
                [(VEHICLE_ID, current_time, speed, temp)]
            )

            # Qdrant
            vector_embedding = [speed/100.0, temp/100.0, random.random(), random.random()]
            qdrant_client.upsert(
                collection_name="vehicle_logs",
                points=[
                    PointStruct(
                        id=int(time.time() * 1000),
                        vector=vector_embedding,
                        payload={"vehicle_id": VEHICLE_ID, "status": "WARNING" if temp > 95 else "NORMAL"}
                    )
                ]
            )

            print(f"[{current_time.strftime('%H:%M:%S')}] 🟢 [{VEHICLE_ID}] Speed: {speed} km/h | Temp: {temp}°C", flush=True)
            
            # Mỗi 10 lần ghi (~20 giây), tự động quét báo cáo 1 lần để xem tốc độ tính toán của ClickHouse
            step += 1
            if step % 10 == 0:
                run_analytics_query()

            time.sleep(2)

    except KeyboardInterrupt:
        print("\n🛑 Đã dừng luồng giả lập.")
    finally:
        pg_conn.close()
        neo4j_driver.close()
        influx_client.close()

if __name__ == "__main__":
    run_simulation()