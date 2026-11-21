"""Étape 5 : interface Streamlit de MaintAssist.

Lancer :  streamlit run app.py
"""
import streamlit as st
from PIL import Image

from src import config
from src.ingestion import indexed_documents

st.set_page_config(page_title="MaintAssist – RAG multimodal agentique", page_icon="🛠️", layout="wide")


# ----------------------------------------------------------------------------- ressources (chargées une fois)
@st.cache_resource(show_spinner="Chargement des modèles et connexion à la base…")
def load_rag(backend: str):
    from src.rag import RAGPipeline
    return RAGPipeline(backend)


@st.cache_resource(show_spinner="Initialisation de l'agent LangChain…")
def load_agent(backend: str):
    from src.agent import MaintenanceAgent
    return MaintenanceAgent(backend)


def run_indexing(pdf_path, backends, describe, vector_pages):
    from src.indexer import index_pdf
    bar = st.progress(0.0, text="Démarrage…")
    stats = index_pdf(pdf_path, backends, describe=describe, render_vector_pages=vector_pages,
                      progress=lambda msg, f: bar.progress(f, text=msg))
    bar.empty()
    st.success(f"✅ {stats['source']} indexé en {stats['seconds']} s : {stats['text']} passages, "
               f"{stats['table']} morceaux de tableaux, {stats['image']} images.")


# ----------------------------------------------------------------------------- barre latérale
with st.sidebar:
    st.title("🛠️ MaintAssist")
    st.caption("Assistant documentaire de maintenance – RAG multimodal agentique")

    backend = st.selectbox("Base vectorielle", config.BACKENDS,
                           index=config.BACKENDS.index(config.DEFAULT_BACKEND))
    mode = st.radio("Mode", ["Agent (LangChain)", "RAG simple"],
                    help="L'agent choisit ses outils de recherche ; le RAG simple interroge toujours texte + tableaux + images.")

    st.divider()
    st.subheader("📄 Documents")
    uploaded = st.file_uploader("Charger un PDF", type=["pdf"])
    existing = sorted(p.name for p in config.PDF_DIR.glob("*.pdf"))
    chosen = st.selectbox("…ou choisir un PDF du dossier pdfs/", ["—"] + existing)
    idx_backends = st.multiselect("Indexer dans", config.BACKENDS, default=[backend])
    describe = st.checkbox("Décrire les images (modèle vision)", value=True)
    vector_pages = st.checkbox("Capturer les schémas vectoriels (manuels officiels)", value=False)
    if st.button("⚙️ Indexer le document", use_container_width=True):
        target = None
        if uploaded is not None:
            target = config.PDF_DIR / uploaded.name
            target.write_bytes(uploaded.getbuffer())
        elif chosen != "—":
            target = config.PDF_DIR / chosen
        if target is None:
            st.warning("Choisissez ou chargez un PDF.")
        elif not idx_backends:
            st.warning("Choisissez au moins une base.")
        else:
            try:
                run_indexing(target, idx_backends, describe, vector_pages)
            except Exception as ex:
                st.error(f"Erreur d'indexation : {ex}")

    if st.button(f"📚 Indexer tous les PDF du dossier ({len(existing)})", use_container_width=True,
                 disabled=not existing or not idx_backends):
        from src.indexer import index_all
        bar = st.progress(0.0, text="Démarrage…")
        results = index_all(idx_backends, describe, vector_pages, progress=lambda m, f: bar.progress(f, text=m))
        bar.empty()
        for r in results:
            if "error" in r:
                st.error(f"❌ {r['source']} : {r['error']}")
            else:
                st.success(f"✅ {r['source']} : {r['text']} txt / {r['table']} tab / {r['image']} img "
                           f"({r['seconds']} s)")

    docs = indexed_documents()
    if docs:
        st.caption("Documents indexés :")
        for d in docs.values():
            st.caption(f"• {d['source']} — {d['text']} txt / {d['table']} tab / {d['image']} img")

    st.divider()
    photo = st.file_uploader("📷 Photo (optionnelle) pour une recherche par image", type=["png", "jpg", "jpeg"])
    if st.button("🗑️ Effacer la conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ----------------------------------------------------------------------------- affichage d'une réponse
def render_extras(res: dict):
    images = res.get("images") or []
    if images:
        st.markdown("**🖼️ Images pertinentes**")
        cols = st.columns(min(3, len(images)))
        for col, img in zip(cols * 3, images):
            with col:
                try:
                    st.image(str(config.abs_path(img["image_path"])), use_container_width=True,
                             caption=f"Page {img['page']} – {img['doc']}")
                except Exception:
                    st.caption(f"(image introuvable : {img.get('image_path')})")
                with st.expander("Description"):
                    st.write(img["content"])
    tables = res.get("tables") or []
    if tables:
        with st.expander(f"📊 Tableaux pertinents ({len(tables)})"):
            for t in tables:
                st.caption(f"Page {t['page']} – {t['doc']}")
                st.markdown(t["content"])
    sources = res.get("sources") or []
    if sources:
        with st.expander(f"🔗 Références ({len(sources)})"):
            for s in sources:
                score = f" – score {s['score']:.2f}" if s.get("score") is not None else ""
                st.markdown(f"- **{s['type']}** · page **{s['page']}** · {s['doc']}{score}")
    steps = res.get("steps") or []
    if steps:
        with st.expander(f"🤖 Raisonnement de l'agent ({len(steps)} appels d'outils)"):
            for i, s in enumerate(steps, 1):
                st.markdown(f"**{i}. `{s['tool']}`** — `{s['args']}`")
                st.code(s["result"][:1200] or "(pas de résultat)", language="markdown")


# ----------------------------------------------------------------------------- conversation
st.header("Posez votre question de maintenance")
st.caption(f"Base : **{backend}** · Mode : **{mode}** · LLM : `{config.LLM_MODEL}` · "
           f"Embeddings : `{config.TEXT_EMB_MODEL}` + CLIP")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("extras"):
            render_extras(msg["extras"])

if not docs:
    st.info("Commencez par indexer un PDF depuis la barre latérale (ex. ABB_ACS580_Guide_Variateur.pdf).")

question = st.chat_input("Ex. : Que signifie le défaut 2310 et comment le corriger ?")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
        if photo is not None:
            st.image(photo, width=220, caption="Photo jointe")

    query_image = Image.open(photo).convert("RGB") if photo is not None else None
    with st.chat_message("assistant"):
        with st.spinner("Recherche dans les manuels et génération de la réponse…"):
            try:
                if mode.startswith("Agent"):
                    history = [{"role": m["role"], "content": m["content"]}
                               for m in st.session_state.messages[:-1]][-6:]
                    res = load_agent(backend).ask(question, history=history, query_image=query_image)
                else:
                    res = load_rag(backend).ask(question, query_image=query_image)
            except Exception as ex:
                res = {"answer": f"❌ Erreur : {ex}\n\nVérifiez qu'Ollama et la base `{backend}` sont démarrés "
                                 f"et que le document est indexé dans cette base."}
        st.markdown(res["answer"])
        render_extras(res)
    st.session_state.messages.append({"role": "assistant", "content": res["answer"],
                                      "extras": {k: v for k, v in res.items() if k != "answer"}})
