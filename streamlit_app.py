import streamlit as st
import sqlite3
import json
import math
from foundry_local_sdk import Configuration, FoundryLocalManager

DB_PATH = "veritabani.db"
CHAT_MODEL = "phi-3.5-mini"
EMBEDDING_MODEL = "qwen3-embedding-0.6b"
MIN_SIMILARITY = 0.45   # Bu değerin altındakileri alma


def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def get_top_chunks(soru_embedding, k=5, min_score=MIN_SIMILARITY):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT metin, embedding FROM parcalar")
    tum_kayitlar = cursor.fetchall()
    conn.close()

    skorlu_liste = []
    for metin, embedding_json in tum_kayitlar:
        embedding = json.loads(embedding_json)
        skor = cosine_similarity(soru_embedding, embedding)
        if skor >= min_score:
            skorlu_liste.append((skor, metin))

    skorlu_liste.sort(key=lambda x: x[0], reverse=True)
    return skorlu_liste[:k]   # (skor, metin) olarak döndür


def cevap_uret(chat_client, soru, baglam_parcalari):
    if not baglam_parcalari:
        return "Bu konuda elimde yeterli bilgi yok."

    baglam = "\n\n".join([metin for skor, metin in baglam_parcalari])

    system_mesaj = (
        "Sen deneyimli bir astroloji asistanısın. "
        "SADECE aşağıda verilen BAĞLAM içindeki bilgilere dayanarak cevap ver. "
        "Bağlamda olmayan hiçbir bilgi uydurma, tahmin etme veya genel bilgi ekleme. "
        "Eğer sorunun cevabı bağlamda yoksa veya çok yetersizse, "
        "sadece şu cümleyi yaz: 'Bu konuda elimde bilgi yok'. "
        "Cevabını Türkçe, kısa, net ve anlaşılır ver."
    )

    kullanici_mesaj = f"BAĞLAM:\n{baglam}\n\nSORU: {soru}"

    response = chat_client.complete_chat([
        {"role": "system", "content": system_mesaj},
        {"role": "user", "content": kullanici_mesaj}
    ])
    return response.choices[0].message.content


@st.cache_resource
def modelleri_yukle():
    config = Configuration(app_name="rag_streamlit")
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


# ---------------- Streamlit Arayüzü ----------------
st.set_page_config(page_title="Astroloji RAG Asistanı", page_icon="✨", layout="centered")
st.title("✨ Astroloji RAG Asistanı")
st.caption("Burçlar, gezegenler, evler ve açılar hakkında sorular sorabilirsiniz. Tüm işlemler bilgisayarında çalışır.")

with st.spinner("Modeller yükleniyor (ilk seferinde birkaç dakika sürebilir)..."):
    embed_client, chat_client = modelleri_yukle()

soru = st.text_input("Sorunuzu yazın:", placeholder="Örn: Akrep burcunun özellikleri nelerdir?")

if st.button("Cevapla", type="primary") and soru.strip():
    with st.spinner("Cevap hazırlanıyor..."):
        response = embed_client.generate_embedding(soru)
        soru_embedding = response.data[0].embedding

        baglam_parcalari = get_top_chunks(soru_embedding, k=5)

        cevap = cevap_uret(chat_client, soru, baglam_parcalari)

    st.subheader("Cevap")
    st.write(cevap)

    if baglam_parcalari:
        with st.expander("Kullanılan bağlam parçaları (benzerlik skorları ile)"):
            for i, (skor, parca) in enumerate(baglam_parcalari):
                st.markdown(f"**{i+1}. (Skor: {skor:.3f})**")
                st.write(parca)
                st.divider()
    else:
        st.info("Yeterince alakalı belge parçası bulunamadı.")