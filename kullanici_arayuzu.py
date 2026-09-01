"""
KULLANICI ARAYÜZÜ (3/3)
=========================
Streamlit tabanlı sohbet arayüzü. Bu dosya HERHANGİ bir retrieval/
generation mantığı içermez; tüm iş `yapay_zeka_motoru.py` modülünde
yapılır. Bu dosyanın tek işi: kullanıcıdan soru almak, motoru
çağırmak ve sonucu güzelce göstermek.

Çalıştırma:
    python -m streamlit run kullanici_arayuzu.py
"""

import streamlit as st

from yapay_zeka_motoru import modelleri_yukle, soru_sor


# --------------------------------------------------
# MODELLERİ ÖNBELLEKLE (sadece ilk çalıştırmada yüklenir)
# --------------------------------------------------

@st.cache_resource
def get_models():
    return modelleri_yukle()


# --------------------------------------------------
# ARAYÜZ
# --------------------------------------------------

st.set_page_config(
    page_title="Astroloji RAG Asistanı",
    page_icon="✨",
    layout="centered"
)

st.title("✨ Astroloji RAG Asistanı")
st.caption(
    "Burçlar, gezegenler, evler ve açılar hakkında sorular sorabilirsiniz. "
    "Tüm işlemler bilgisayarında, tamamen yerel olarak çalışır."
)

with st.spinner("Modeller yükleniyor (ilk seferinde birkaç dakika sürebilir)..."):
    embed_client, chat_client = get_models()

soru = st.text_input(
    "Sorunuzu yazın:",
    placeholder="Örn: 7. evde Jüpiter ne anlama gelir?"
)

if st.button("Cevapla", type="primary") and soru.strip():
    with st.spinner("Cevap hazırlanıyor..."):
        cevap, baglam_parcalari = soru_sor(embed_client, chat_client, soru, k=1)

    st.subheader("Cevap")
    st.write(cevap)

    if baglam_parcalari:
        with st.expander("Kullanılan bağlam parçaları"):
            for i, (final_score, semantic_score, exact_score, parca) in enumerate(baglam_parcalari, start=1):
                st.markdown(
                    f"**{i}.** Final: `{final_score:.3f}` | "
                    f"Semantic: `{semantic_score:.3f}` | "
                    f"Keyword: `{exact_score:.3f}`"
                )
                st.write(parca)
                st.divider()
    else:
        st.info("Yeterince alakalı belge parçası bulunamadı.")
