import sqlite3
import json
import os
from foundry_local_sdk import Configuration, FoundryLocalManager

DB_PATH = "veritabani.db"
BELGE_PATH = "belgeler/burclar.txt"
EMBEDDING_MODEL = "qwen3-embedding-0.6b"
MAX_CHUNK_CHARS = 800   # Çok uzun paragrafları böl


def veritabani_olustur():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS parcalar (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metin TEXT NOT NULL,
            embedding TEXT NOT NULL,
            kaynak TEXT
        )
    """)
    conn.commit()
    return conn


def metni_parcalara_ayir(metin: str):
    """Önce boş satırlara göre böl, sonra çok uzun olanları daha küçük parçalara ayır."""
    ham_parcalar = [p.strip() for p in metin.split("\n\n") if p.strip()]
    parcalar = []

    for parca in ham_parcalar:
        if len(parca) <= MAX_CHUNK_CHARS:
            parcalar.append(parca)
        else:
            # Uzun paragrafı cümle sonlarına göre bölmeye çalış
            cumleler = parca.replace(". ", ".\n").split("\n")
            mevcut = ""
            for cumle in cumleler:
                if len(mevcut) + len(cumle) + 1 <= MAX_CHUNK_CHARS:
                    mevcut = (mevcut + " " + cumle).strip()
                else:
                    if mevcut:
                        parcalar.append(mevcut)
                    mevcut = cumle
            if mevcut:
                parcalar.append(mevcut)

    return parcalar


def main():
    if not os.path.exists(BELGE_PATH):
        print(f"Hata: '{BELGE_PATH}' dosyası bulunamadı.")
        return

    print("Belgeler okunuyor...")
    with open(BELGE_PATH, "r", encoding="utf-8") as f:
        icerik = f.read()

    parcalar = metni_parcalara_ayir(icerik)
    print(f"{len(parcalar)} parça bulundu.")

    if len(parcalar) == 0:
        print("Hiç parça bulunamadı. Belge boş olabilir.")
        return

    print("Embedding modeli yükleniyor...")
    config = Configuration(app_name="rag_ingest")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    model = manager.catalog.get_model(EMBEDDING_MODEL)
    model.download()
    model.load()
    client = model.get_embedding_client()

    conn = veritabani_olustur()
    cursor = conn.cursor()

    # Eski verileri temizle
    cursor.execute("DELETE FROM parcalar")

    print("Parçalar embedding'leniyor ve kaydediliyor...")
    for i, parca in enumerate(parcalar):
        response = client.generate_embedding(parca)
        embedding = response.data[0].embedding
        embedding_json = json.dumps(embedding)

        cursor.execute(
            "INSERT INTO parcalar (metin, embedding, kaynak) VALUES (?, ?, ?)",
            (parca, embedding_json, "burclar.txt")
        )
        print(f"  {i+1}/{len(parcalar)} tamamlandı")

    conn.commit()
    conn.close()
    model.unload()

    print(f"\nTamamlandı! {len(parcalar)} parça '{DB_PATH}' dosyasına kaydedildi.")


if __name__ == "__main__":
    main()