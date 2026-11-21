# MaintAssist – RAG multimodal agentique pour la maintenance (Partie I)

Assistant documentaire pour techniciens de maintenance. Il est construit sur la documentation du **variateur de fréquence ABB ACS580**.
Il lit un PDF technique (texte, tableaux et images), l'indexe dans **PGVector**, **Qdrant** et **Chroma**, puis répond aux questions grâce à un **agent LangChain** qui utilise **Qwen3.5 9B** (via Ollama). L'interface est faite avec **Streamlit**.

```
PDF ──► LlamaIndex + PyMuPDF + pdfplumber ──► texte (chunks) / tableaux (Markdown) / images (PNG)
                                                    │                    │
                                       bge-m3 (texte, tableaux,     qwen3.5 (description) 
                                       descriptions d'images)       + CLIP (embedding image)
                                                    ▼                    ▼
                                   PGVector (PostgreSQL) · Qdrant · Chroma DB
                                                    ▼
             Agent LangChain (Qwen3.5 9B / Ollama) ── outils : code défaut · texte · tableaux · images
                                                    ▼
                         Streamlit : réponse + images + tableaux + pages sources
```

## Correspondance avec le travail demandé

| # | Travail demandé | Où dans le projet |
|---|---|---|
| 1 | Sélection du document | `pdfs/ABB_ACS580_Guide_Variateur.pdf` (+ manuels officiels ABB) |
| 2 | Extraction et indexation (LlamaIndex, embeddings, PGVector/Qdrant/Chroma) | `src/ingestion.py`, `src/embeddings.py`, `src/vectorstore.py`, `src/indexer.py` · notebooks 01 et 03 |
| 3 | Traitement des images (CLIP + descriptions) | `src/images.py`, `src/embeddings.py` · notebook 02 |
| 4 | Question-réponse avec un LLM open source (Qwen/Ollama) | `src/rag.py` · notebook 04 |
| 5 | Interface Streamlit | `app.py` |
| 6 | Orchestration agentique LangChain | `src/agent.py` · notebook 05 |

---

## Guide pas à pas (Windows)

### Étape 0 – Installer les outils (une seule fois)
1. **Python 3.11** (3.10 à 3.12 fonctionnent) : https://www.python.org. Coche « Add Python to PATH » pendant l'installation.
2. **Docker Desktop** : https://www.docker.com/products/docker-desktop. Lance-le et attends que l'icône indique « running ».
3. **Ollama** : https://ollama.com/download.
4. **VS Code** avec les extensions *Python* et *Jupyter*.

### Étape 1 – Créer l'environnement Python
Ouvre un terminal PowerShell **dans le dossier du projet** :
```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
# PyTorch avec CUDA (pour la RTX) – à installer AVANT le reste
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
copy .env.example .env
```
Pour vérifier que le GPU est détecté :
```powershell
python -c "import torch; print(torch.cuda.is_available())"
```
La commande doit afficher `True`.

### Étape 2 – Démarrer les bases vectorielles
```powershell
docker compose up -d
docker ps
```
`docker ps` doit afficher `maintassist-pgvector` et `maintassist-qdrant`. Le tableau de bord de Qdrant est accessible sur http://localhost:6333/dashboard.
Chroma n'a pas besoin de Docker : ses données sont stockées dans `data/chroma`.

### Étape 3 – Modèle Ollama
```powershell
ollama pull qwen3.5:9b      # 6,6 Go – multimodal : génération, appels d'outils ET description des images
ollama list                 # vérifier qu'il apparaît
```
Un seul modèle sert pour tout. Ollama n'a donc pas besoin de basculer entre deux modèles, ce qui fait gagner du temps.
Le mode « réflexion » de Qwen3.5 est désactivé par défaut (`LLM_THINK=false` dans `.env`). Mets `true` pour comparer pendant la démo.
Les modèles d'embeddings (`BAAI/bge-m3` ≈ 2,3 Go et CLIP) se téléchargent automatiquement depuis Hugging Face au premier lancement.

### Étape 4 – Exécuter les notebooks dans l'ordre
Ouvre le dossier dans VS Code. Pour chaque notebook, choisis le kernel **.venv**.

| Notebook | Ce qu'il montre |
|---|---|
| `01_ingestion.ipynb` | Extraction : nombre de chunks, tableaux en Markdown, images extraites |
| `02_images_clip.ipynb` | Descriptions générées par qwen3.5:9b (vision), matrice de similarité CLIP texte→image, recherche image→image |
| `03_vectorstores.ipynb` | Indexation dans les 3 bases et benchmark latence / hit@5 |
| `04_rag.ipynb` | Réponses du RAG avec sources et images |
| `05_agent.ipynb` | Agent LangChain : outils choisis, réponses, mémoire de conversation |

Tu peux aussi indexer en ligne de commande :
```powershell
python -m src.indexer pdfs/ABB_ACS580_Guide_Variateur.pdf --backends pgvector qdrant chroma
```

### Étape 5 – Lancer l'application
```powershell
streamlit run app.py
```
Dans la barre latérale :
1. Choisis la base vectorielle.
2. Choisis un PDF et clique sur **Indexer**.
3. Pose tes questions.

L'option « Photo » permet une recherche par image avec CLIP.

### Questions pour la démo
- « Le variateur affiche 2310, que dois-je faire ? » : l'agent fait une recherche exacte du code, puis cherche dans les procédures et les logigrammes.
- « Montre-moi le schéma de câblage de la macro ABB standard » : la réponse affiche l'image du bornier.
- « Tous les combien faut-il changer le ventilateur d'un ACS580 en taille R3 ? » : la réponse vient d'un tableau de maintenance.
- « Différence entre les défauts 2330 et 3181 ? » : l'agent fait plusieurs appels d'outils.
- « Le panneau affiche 5091 puis FA81, est-ce une panne ? » : la réponse porte sur la fonction de sécurité STO.

---

## Les manuels officiels ABB
Ces PDF font plusieurs centaines de pages et leurs schémas sont **vectoriels**, pas des images.
- Coche « Capturer les schémas vectoriels » dans l'application, ou ajoute l'option `--vector-pages` en ligne de commande. Les pages de schémas seront alors rendues en images.
- Pour gagner du temps, extrais d'abord un chapitre, par exemple *Fault tracing* :
```powershell
python -c "import pymupdf; d=pymupdf.open('pdfs/EN_ACS580_FW_J_A5.pdf'); n=pymupdf.open(); n.insert_pdf(d, from_page=500, to_page=560); n.save('pdfs/ACS580_fault_tracing.pdf')"
```
Adapte les numéros de pages à la table des matières du manuel.

## Dépannage
| Problème | Solution |
|---|---|
| `connection refused` sur le port 5433 ou 6333 | Lance Docker Desktop, puis `docker compose up -d`. |
| `Ollama … connection refused` | Lance l'application Ollama, ou `ollama serve`. |
| Manque de mémoire GPU (CUDA out of memory) | Mets `EMB_DEVICE=cpu` dans `.env`. Ollama décharge aussi automatiquement une partie du modèle sur le CPU si la VRAM manque (plus lent). Ferme les autres applications qui utilisent le GPU. |
| `different vector dimensions` | Tu as changé de modèle d'embeddings. Vide les bases avec `docker compose down -v` et supprime `data/chroma`, puis indexe à nouveau. |
| Réponse « aucune image pertinente » | Baisse `IMG_DESC_MIN_SCORE` ou `IMG_CLIP_MIN_SCORE` dans `.env`. |
| Le premier lancement est lent | C'est normal : les modèles se téléchargent. Les descriptions d'images sont ensuite mises en cache dans `data/cache`. |

## Structure
```
├── app.py                 Interface Streamlit
├── docker-compose.yml     PostgreSQL+pgvector, Qdrant
├── requirements.txt / .env.example
├── pdfs/                  Documents sources
├── data/                  (généré) images, éléments JSON, cache, chroma
├── notebooks/             01 → 05
└── src/
    ├── config.py          Paramètres (.env)
    ├── ingestion.py       Parsing PDF (LlamaIndex, PyMuPDF, pdfplumber)
    ├── images.py          Descriptions d'images (qwen3.5 / Ollama)
    ├── embeddings.py      bge-m3 + CLIP
    ├── vectorstore.py     PGVector / Qdrant / Chroma
    ├── indexer.py         Pipeline d'indexation (+ CLI)
    ├── rag.py             Recherche multimodale + génération
    └── agent.py           Agent LangChain + outils
```
#   p a r t i 1 _ A g e n t i c P r o j e c t  
 #   p a r t i 1 _ A g e n t i c P r o j e c t  
 