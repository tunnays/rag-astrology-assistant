"""
YAPAY ZEKA MOTORU (2/3)
========================
RAG hattının beyni. Bu dosya herhangi bir arayüz (Streamlit, CLI, vb.)
bilgisi içermez; sadece şu iki işi yapar:

    1) RETRIEVAL: Soruya en alakalı belge parçalarını bulmak
       - Semantic similarity (embedding + cosine similarity)
       - Keyword/exact matching (burç, gezegen, ev numarası eşleşmesi)
       - final_score = semantic_score * 0.70 + keyword_score * 0.30

    2) GENERATION: Bulunan parçalara dayanarak cevap üretmek
       - Çok güçlü bir exact match varsa kaynak metni doğrudan döndürür
         (halüsinasyon riskini azaltır, daha hızlıdır)
       - Aksi halde phi-3.5-mini modeline bağlamı vererek cevap ürettirir

Bu modül `kullanici_arayuzu.py` tarafından import edilerek kullanılır.
Terminalden bağımsız test etmek için de doğrudan çalıştırılabilir:

    python yapay_zeka_motoru.py
"""

import sqlite3
import json
import math
import re

from foundry_local_sdk import Configuration, FoundryLocalManager


# --------------------------------------------------
# YAPILANDIRMA
# --------------------------------------------------

DB_PATH = "veritabani.db"
EMBEDDING_MODEL = "qwen3-embedding-0.6b"
CHAT_MODEL = "phi-3.5-mini"

MIN_SIMILARITY = 0.45   # Bu eşiğin altındaki semantic skorlar elenir
TOP_K = 5                # Varsayılan olarak döndürülecek parça sayısı
EXACT_MATCH_ESIGI = 0.60  # Bu eşiğin üzerinde LLM'e gitmeden kaynak metin döndürülür


# ==================================================
# BÖLÜM A: RETRIEVAL (retrieval.py'den taşındı)
# ==================================================

def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)          # Noktalama işaretlerini kaldır
    text = re.sub(r"\s+", " ", text).strip()       # Fazla boşlukları temizle
    return text


def ev_numarasini_bul(text):
    """'7. ev', '7 evde', 'yedinci ev' gibi ifadelerden ev numarasını çıkarır."""
    text = normalize_text(text)

    # Rakamla yazılan evler: 7. ev, 7 ev, 7. evde gibi
    match = re.search(r"\b(1[0-2]|[1-9])\s*ev(?:de|da|in|ın|un|ün)?\b", text)
    if match:
        return match.group(1)

    # Yazıyla yazılan evler: yedinci ev, on birinci ev vb.
    evler = {
        "birinci": "1", "ikinci": "2", "üçüncü": "3", "dördüncü": "4",
        "beşinci": "5", "altıncı": "6", "yedinci": "7", "sekizinci": "8",
        "dokuzuncu": "9", "onuncu": "10", "on birinci": "11", "on ikinci": "12"
    }
    for kelime, numara in evler.items():
        if re.search(rf"\b{re.escape(kelime)}\s+ev", text):
            return numara

    return None


def keyword_score(soru, metin):
    """Burç/gezegen adı ve ev numarası eşleşmesine göre 0.0-0.65 arası bir skor üretir."""
    soru_normal = normalize_text(soru)
    metin_normal = normalize_text(metin)

    skor = 0.0
    gezegenler = [
        "güneş", "ay", "merkür", "venüs", "mars",
        "jüpiter", "satürn", "uranüs", "neptün", "plüton"
    ]

    bulunan_gezegen = None
    for gezegen in gezegenler:
        if gezegen in soru_normal:
            bulunan_gezegen = gezegen
            if gezegen in metin_normal:
                skor += 0.15
            break

    bulunan_ev = ev_numarasini_bul(soru)
    if bulunan_ev:
        ev_pattern = rf"\b{bulunan_ev}\s*ev(?:de|da)?\b"
        if re.search(ev_pattern, metin_normal):
            skor += 0.20

    if bulunan_gezegen and bulunan_ev:
        tam_eslesme = rf"\b{bulunan_ev}\s*ev(?:de|da)?\s+{bulunan_gezegen}\b"
        if re.search(tam_eslesme, metin_normal):
            skor += 0.30

    return skor


def get_top_chunks(soru, soru_embedding, k=TOP_K, min_score=MIN_SIMILARITY):
    """
    Hybrid retrieval: semantic similarity + keyword score birleştirilerek
    en alakalı k parça döndürülür.

    Dönüş: [(final_score, semantic_score, exact_score, metin), ...]
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT metin, embedding FROM parcalar")
    tum_kayitlar = cursor.fetchall()
    conn.close()

    skorlu_liste = []
    for metin, embedding_json in tum_kayitlar:
        embedding = json.loads(embedding_json)

        semantic_score = cosine_similarity(soru_embedding, embedding)
        exact_score = keyword_score(soru, metin)
        final_score = semantic_score * 0.70 + exact_score * 0.30

        if semantic_score >= min_score:
            skorlu_liste.append((final_score, semantic_score, exact_score, metin))

    skorlu_liste.sort(key=lambda x: x[0], reverse=True)
    return skorlu_liste[:k]


# ==================================================
# BÖLÜM B: GENERATION (rag_app.py'den taşındı)
# ==================================================

def cevap_uret(chat_client, soru, baglam_parcalari):
    """
    Bulunan bağlam parçalarına göre cevap üretir.
    Güçlü bir exact match varsa LLM'e hiç sormadan kaynak metni döndürür.
    """
    if not baglam_parcalari:
        return "Bu konuda elimde bilgi yok."

    final_score, semantic_score, exact_score, en_iyi_metin = baglam_parcalari[0]

    # Çok güçlü exact match varsa LLM'e göndermeden doğrudan döndür
    if exact_score >= EXACT_MATCH_ESIGI:
        return en_iyi_metin

    baglam = "\n\n".join(metin for _, _, _, metin in baglam_parcalari)

    system_mesaj = (
        "Sen bir bilgi çıkarma asistanısın. "
        "Yalnızca verilen BAĞLAM içindeki bilgileri kullan. "
        "Bağlamda cevap varsa mutlaka bu bilgiyi kullanarak cevap ver. "
        "Bağlamdaki anlamı değiştirme ve yeni bilgi ekleme. "
        "Bağlamda cevap gerçekten yoksa sadece "
        "'Bu konuda elimde bilgi yok.' yaz. "
        "Türkçe ve kısa cevap ver."
    )

    kullanici_mesaj = f"""
BAĞLAM:
----------------
{baglam}
----------------

SORU:
{soru}

CEVAP:
"""

    response = chat_client.complete_chat([
        {"role": "system", "content": system_mesaj},
        {"role": "user", "content": kullanici_mesaj}
    ])

    return response.choices[0].message.content


# ==================================================
# BÖLÜM C: MODEL YÜKLEME
# ==================================================

def modelleri_yukle():
    """
    Foundry Local üzerinden embedding ve chat modellerini yükler.
    Not: Bu fonksiyon Streamlit'e bağımlı değildir; arayüz katmanı
    isterse kendi önbellekleme (st.cache_resource) mekanizmasını
    bunun etrafına sarabilir.
    """
    config = Configuration(app_name="rag_engine")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    embed_model = manager.catalog.get_model(EMBEDDING_MODEL)
    embed_model.download()
    embed_model.load()
    embed_client = embed_model.get_embedding_client()

    chat_model = manager.catalog.get_model(CHAT_MODEL)
    chat_model.download()
    chat_model.load()
    chat_client = chat_model.get_chat_client()

    return embed_client, chat_client


# ==================================================
# BÖLÜM D: TEK ADIMDA SORU-CEVAP (arayüzün kullandığı ana fonksiyon)
# ==================================================

def soru_sor(embed_client, chat_client, soru, k=1):
    """
    Kullanıcı arayüzünün çağıracağı tek giriş noktası.
    Soru -> embedding -> retrieval -> generation -> (cevap, bağlam) döndürür.
    """
    response = embed_client.generate_embedding(soru)
    soru_embedding = response.data[0].embedding

    baglam_parcalari = get_top_chunks(soru, soru_embedding, k=k)
    cevap = cevap_uret(chat_client, soru, baglam_parcalari)

    return cevap, baglam_parcalari


# --------------------------------------------------
# TERMİNALDEN BAĞIMSIZ TEST
# --------------------------------------------------

def main():
    print("Modeller yükleniyor...\n")
    embed_client, chat_client = modelleri_yukle()

    soru = input("Sorunuzu yazın: ").strip()
    if not soru:
        print("Boş soru girdiniz.")
        return

    print("\nCevap hazırlanıyor...\n")
    cevap, baglam_parcalari = soru_sor(embed_client, chat_client, soru, k=TOP_K)

    print("=== CEVAP ===")
    print(cevap)
    print()

    if baglam_parcalari:
        print("=== KULLANILAN BAĞLAM PARÇALARI ===")
        for i, (final_score, semantic_score, exact_score, metin) in enumerate(baglam_parcalari, start=1):
            print(f"[{i}] Final: {final_score:.3f} | Semantic: {semantic_score:.3f} | Keyword: {exact_score:.3f}")
            print(metin)
            print("-" * 60)


if __name__ == "__main__":
    main()
