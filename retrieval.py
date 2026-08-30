import sqlite3
import json
import math
import re

from foundry_local_sdk import Configuration, FoundryLocalManager


DB_PATH = "veritabani.db"
EMBEDDING_MODEL = "qwen3-embedding-0.6b"

MIN_SIMILARITY = 0.45
TOP_K = 5


# --------------------------------------------------
# COSINE SIMILARITY
# --------------------------------------------------

def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))

    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)


# --------------------------------------------------
# METİN NORMALLEŞTİRME
# --------------------------------------------------

def normalize_text(text):
    text = text.lower()

    # Noktalama işaretlerini kaldır
    text = re.sub(r"[^\w\s]", " ", text)

    # Fazla boşlukları temizle
    text = re.sub(r"\s+", " ", text).strip()

    return text


# --------------------------------------------------
# EV NUMARASINI BUL
# --------------------------------------------------

def ev_numarasini_bul(text):
    text = normalize_text(text)

    # Rakamla yazılan evleri bul:
    # 7. ev, 7 ev, 7. evde gibi
    match = re.search(
        r"\b(1[0-2]|[1-9])\s*ev(?:de|da|in|ın|un|ün)?\b",
        text
    )

    if match:
        return match.group(1)

    # Yazıyla yazılan evleri bul:
    # yedinci ev, on birinci ev vb.
    evler = {
        "birinci": "1",
        "ikinci": "2",
        "üçüncü": "3",
        "dördüncü": "4",
        "beşinci": "5",
        "altıncı": "6",
        "yedinci": "7",
        "sekizinci": "8",
        "dokuzuncu": "9",
        "onuncu": "10",
        "on birinci": "11",
        "on ikinci": "12"
    }

    for kelime, numara in evler.items():
        if re.search(
            rf"\b{re.escape(kelime)}\s+ev",
            text
        ):
            return numara

    return None


# --------------------------------------------------
# KEYWORD / EXACT MATCH SKORU
# --------------------------------------------------

def keyword_score(soru, metin):
    soru_normal = normalize_text(soru)
    metin_normal = normalize_text(metin)

    skor = 0.0

    gezegenler = [
        "güneş",
        "ay",
        "merkür",
        "venüs",
        "mars",
        "jüpiter",
        "satürn",
        "uranüs",
        "neptün",
        "plüton"
    ]

    bulunan_gezegen = None

    # --------------------------------------------------
    # 1. GEZEGEN EŞLEŞMESİ
    # --------------------------------------------------

    for gezegen in gezegenler:
        if gezegen in soru_normal:
            bulunan_gezegen = gezegen

            if gezegen in metin_normal:
                skor += 0.15

            break

    # --------------------------------------------------
    # 2. EV EŞLEŞMESİ
    # --------------------------------------------------

    bulunan_ev = ev_numarasini_bul(soru)

    if bulunan_ev:
        ev_pattern = rf"\b{bulunan_ev}\s*ev(?:de|da)?\b"

        if re.search(ev_pattern, metin_normal):
            skor += 0.20

    # --------------------------------------------------
    # 3. TAM GEZEGEN + EV EŞLEŞMESİ
    # --------------------------------------------------

    if bulunan_gezegen and bulunan_ev:
        tam_eslesme = (
            rf"\b{bulunan_ev}\s*ev(?:de|da)?\s+"
            rf"{bulunan_gezegen}\b"
        )

        if re.search(tam_eslesme, metin_normal):
            skor += 0.30

    return skor


# --------------------------------------------------
# TOP CHUNKS
# --------------------------------------------------

def get_top_chunks(
    soru,
    soru_embedding,
    k=TOP_K,
    min_score=MIN_SIMILARITY
):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT metin, embedding FROM parcalar"
    )

    tum_kayitlar = cursor.fetchall()

    conn.close()

    skorlu_liste = []

    for metin, embedding_json in tum_kayitlar:

        embedding = json.loads(embedding_json)

        # --------------------------------------------------
        # 1. SEMANTIC SCORE
        # --------------------------------------------------

        semantic_score = cosine_similarity(
            soru_embedding,
            embedding
        )

        # --------------------------------------------------
        # 2. KEYWORD SCORE
        # --------------------------------------------------

        exact_score = keyword_score(
            soru,
            metin
        )

        # --------------------------------------------------
        # 3. FINAL SCORE
        # --------------------------------------------------

        final_score = (
            semantic_score * 0.70
            +
            exact_score * 0.30
        )

        # --------------------------------------------------
        # Minimum semantic threshold
        # --------------------------------------------------

        if semantic_score >= min_score:
            skorlu_liste.append(
                (
                    final_score,
                    semantic_score,
                    exact_score,
                    metin
                )
            )

    # Final skora göre büyükten küçüğe sırala
    skorlu_liste.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return skorlu_liste[:k]


# --------------------------------------------------
# TEST
# --------------------------------------------------

def main():

    print("Embedding modeli yükleniyor...\n")

    config = Configuration(
        app_name="rag_retrieval"
    )

    FoundryLocalManager.initialize(config)

    manager = FoundryLocalManager.instance

    embed_model = manager.catalog.get_model(
        EMBEDDING_MODEL
    )

    embed_model.download()
    embed_model.load()

    embed_client = embed_model.get_embedding_client()

    soru = input(
        "Sorunuzu yazın: "
    ).strip()

    if not soru:
        print("Boş soru girdiniz.")
        embed_model.unload()
        return

    # --------------------------------------------------
    # SORUYU EMBEDDING'E ÇEVİR
    # --------------------------------------------------

    response = embed_client.generate_embedding(
        soru
    )

    soru_embedding = response.data[0].embedding

    print(
        "\nEn alakalı parçalar bulunuyor...\n"
    )

    sonuclar = get_top_chunks(
        soru,
        soru_embedding
    )

    if not sonuclar:
        print(
            f"Benzerlik eşiği "
            f"({MIN_SIMILARITY}) üzerinde "
            f"parça bulunamadı."
        )

    else:
        for i, (
            final_score,
            semantic_score,
            exact_score,
            metin
        ) in enumerate(sonuclar, start=1):

            print(
                f"[{i}] "
                f"Final: {final_score:.3f} | "
                f"Semantic: {semantic_score:.3f} | "
                f"Keyword: {exact_score:.3f}"
            )

            print(metin)

            print("-" * 60)
            print()

    embed_model.unload()

    print("Bitti.")


if __name__ == "__main__":
    main()