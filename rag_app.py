import streamlit as st

from foundry_local_sdk import (
    Configuration,
    FoundryLocalManager
)

from retrieval import get_top_chunks


CHAT_MODEL = "phi-3.5-mini"
EMBEDDING_MODEL = "qwen3-embedding-0.6b"


def cevap_uret(chat_client, soru, baglam_parcalari):

    if not baglam_parcalari:
        return "Bu konuda elimde bilgi yok."

    # En iyi parçayı al
    final_score, semantic_score, exact_score, en_iyi_metin = baglam_parcalari[0]

    # Çok güçlü exact match varsa LLM'e göndermeden doğrudan döndür
    if exact_score >= 0.60:
        return en_iyi_metin

    # Diğer sorularda LLM kullanalım
    baglam = "\n\n".join(
        metin
        for final_score, semantic_score, exact_score, metin
        in baglam_parcalari
    )

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
        {
            "role": "system",
            "content": system_mesaj
        },
        {
            "role": "user",
            "content": kullanici_mesaj
        }
    ])

    return response.choices[0].message.content

# --------------------------------------------------
# MODELLER
# --------------------------------------------------

@st.cache_resource
def modelleri_yukle():

    config = Configuration(
        app_name="rag_streamlit"
    )

    FoundryLocalManager.initialize(
        config
    )

    manager = FoundryLocalManager.instance

    # ------------------------------------------
    # Embedding model
    # ------------------------------------------

    embed_model = manager.catalog.get_model(
        EMBEDDING_MODEL
    )

    embed_model.download()
    embed_model.load()

    embed_client = (
        embed_model.get_embedding_client()
    )

    # ------------------------------------------
    # Chat model
    # ------------------------------------------

    chat_model = manager.catalog.get_model(
        CHAT_MODEL
    )

    chat_model.download()
    chat_model.load()

    chat_client = (
        chat_model.get_chat_client()
    )

    return (
        embed_client,
        chat_client
    )


# --------------------------------------------------
# STREAMLIT
# --------------------------------------------------

st.set_page_config(
    page_title="Astroloji RAG Asistanı",
    page_icon="✨",
    layout="centered"
)

st.title(
    "✨ Astroloji RAG Asistanı"
)

st.caption(
    "Burçlar, gezegenler, evler ve açılar "
    "hakkında sorular sorabilirsiniz. "
    "Tüm işlemler bilgisayarında çalışır."
)


with st.spinner(
    "Modeller yükleniyor "
    "(ilk seferinde birkaç dakika sürebilir)..."
):

    embed_client, chat_client = (
        modelleri_yukle()
    )


soru = st.text_input(
    "Sorunuzu yazın:",
    placeholder=(
        "Örn: 7. evde Jüpiter ne anlama gelir?"
    )
)


if (
    st.button(
        "Cevapla",
        type="primary"
    )
    and soru.strip()
):

    with st.spinner(
        "Cevap hazırlanıyor..."
    ):

        # --------------------------------------
        # Soru embedding
        # --------------------------------------

        response = (
            embed_client.generate_embedding(
                soru
            )
        )

        soru_embedding = (
            response.data[0].embedding
        )

        # --------------------------------------
        # Retrieval
        # --------------------------------------

        baglam_parcalari = get_top_chunks(
    soru,
    soru_embedding,
    k=1
)
        # --------------------------------------
        # LLM
        # --------------------------------------

        cevap = cevap_uret(
            chat_client,
            soru,
            baglam_parcalari
        )

    # ------------------------------------------
    # CEVAP
    # ------------------------------------------

    st.subheader("Cevap")

    st.write(cevap)

    # ------------------------------------------
    # DEBUG / CONTEXT
    # ------------------------------------------

    if baglam_parcalari:

        with st.expander(
            "Kullanılan bağlam parçaları"
        ):

            for i, (
                final_score,
                semantic_score,
                exact_score,
                parca
            ) in enumerate(
                baglam_parcalari
            ):

                st.markdown(
                    f"**{i+1}.** "
                    f"Final: `{final_score:.3f}` | "
                    f"Semantic: `{semantic_score:.3f}` | "
                    f"Keyword: `{exact_score:.3f}`"
                )

                st.write(parca)

                st.divider()

    else:

        st.info(
            "Yeterince alakalı belge parçası bulunamadı."
        )