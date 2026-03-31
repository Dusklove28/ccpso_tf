import sqlite3
import json
import os


def export_db_to_json():
    # 自动寻找数据库文件（优先找你之前备份的，如果没有就找原名）
    db_file = 'db_backup.db' if os.path.exists('db_backup.db') else 'db.db'

    if not os.path.exists(db_file):
        print(f"❌ 找不到数据库文件 {db_file}")
        return

    print(f"📂 正在读取数据库: {db_file}")
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # 获取所有表名
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    db_dump = {}
    for table_name in tables:
        table_name = table_name[0]
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()

        # 获取列名
        column_names = [description[0] for description in cursor.description]

        table_data = []
        for row in rows:
            # 将每一行与列名组合成字典
            table_data.append(dict(zip(column_names, row)))

        db_dump[table_name] = table_data
        print(f"✅ 成功导出表: {table_name} (共 {len(rows)} 条记录)")

    # 写入 JSON 文件
    output_file = 'db_dump.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(db_dump, f, indent=4, ensure_ascii=False)

    print(f"\n🎉 转换完成！所有数据已保存至 {output_file}")


if __name__ == "__main__":
    export_db_to_json()