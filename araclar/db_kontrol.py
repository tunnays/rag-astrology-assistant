"""
YARDIMCI SCRIPT: Veritabanı Kontrolü
======================================
veri_hazirlama.py çalıştırıldıktan sonra veritabani.db içeriğinin
doğru dolduğunu hızlıca doğrulamak için kullanılır. Proje kökünden
çalıştırılmalıdır:

    python araclar/db_kontrol.py
"""

import sqlite3
import json

DB_PATH = "veritabani.db"


def main():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Toplam kayıt sayısı
    cursor.execute("SELECT COUNT(*) FROM parcalar")
    toplam = cursor.fetchone()[0]
    print(f"Toplam kayıt sayısı: {toplam}\n")

    # İlk 3 kaydı örnek olarak göster
    cursor.execute("SELECT id, metin, embedding FROM parcalar LIMIT 3")
    kayitlar = cursor.fetchall()

    for kayit_id, metin, embedding_json in kayitlar:
        embedding = json.loads(embedding_json)
        print(f"--- Kayıt #{kayit_id} ---")
        print(f"Metin (ilk 100 karakter): {metin[:100]}...")
        print(f"Embedding vektör boyutu: {len(embedding)}")
        print(f"İlk 3 değer: {embedding[:3]}")
        print()

    conn.close()


if __name__ == "__main__":
    main()
