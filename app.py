import streamlit as st
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D
import pickle
import numpy as np
from numpy.linalg import norm

# =========================================================
# 1. CONFIGURATION DE LA PAGE & TITRE
# =========================================================
st.set_page_config(page_title="Word2Vec - Exploration", page_icon="🎬", layout="centered")

st.title("🎬 Exploration des Word Embeddings (Word2Vec)")
st.write("""
Cette application permet de visualiser et d'explorer les relations sémantiques et 
arithmétiques apprises par un modèle **Word2Vec** entraîné sur des critiques de films.
""")

# =========================================================
# 2. PARAMÈTRES DU MODÈLE
# =========================================================
vocab_size = 10000
embedding_dim = 300

# =========================================================
# 3. CHARGEMENT DU MODÈLE ET DES POIDS
# =========================================================
@st.cache_resource
def build_and_load_model():
    # Définition de l'architecture du modèle
    model = Sequential([
        Embedding(vocab_size, embedding_dim),
        GlobalAveragePooling1D(),
        Dense(vocab_size, activation='softmax')
    ])

    # Construction explicite des couches en mémoire
    model.build((None, None))

    # Chargement des poids entraînés
    model.load_weights("word2vec.h5")

    # Chargement du tokenizer
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    return model, tokenizer

model, tokenizer = build_and_load_model()
st.success("✅ Modèle, poids et tokenizer chargés avec succès !")

# =========================================================
# 4. EXTRACTION DE LA MATRICE D'EMBEDDINGS
# =========================================================
@st.cache_resource
def get_embedding_matrix():
    # La première couche (index 0) contient la matrice d'embeddings
    return model.layers[0].get_weights()[0]

embedding_matrix = get_embedding_matrix()

st.caption(f"📊 Dimension de la matrice d'embeddings : **{embedding_matrix.shape[0]} mots** × **{embedding_matrix.shape[1]} dimensions**")

# =========================================================
# 5. FONCTIONS DE SIMILITUDE
# =========================================================
def cosine_similarity(vec_a, vec_b):
    """Calcule la similarité cosinus entre deux vecteurs."""
    dot_product = np.dot(vec_a, vec_b)
    norm_a = norm(vec_a)
    norm_b = norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))

def get_word_vector(word):
    """Retourne le vecteur d'embedding pour un mot donné."""
    word = word.lower().strip()
    if word in tokenizer.word_index:
        idx = tokenizer.word_index[word]
        if idx < vocab_size:
            return embedding_matrix[idx]
    return None

def print_closest(word, top_n=10):
    """Trouve et affiche les N mots les plus proches d'un mot cible."""
    vec = get_word_vector(word)
    if vec is None:
        st.warning(f"⚠️ Le mot **'{word}'** n'est pas présent dans le vocabulaire (Top {vocab_size}).")
        return

    similarities = []
    for w, idx in tokenizer.word_index.items():
        if idx < vocab_size and w != word.lower().strip():
            sim = cosine_similarity(vec, embedding_matrix[idx])
            similarities.append((w, sim))

    # Tri décroissant selon la similarité
    similarities.sort(key=lambda x: x[1], reverse=True)
    closest = similarities[:top_n]

    st.subheader(f"🔎 Les {top_n} mots les plus proches de : *'{word}'*")

    for rank, (w, sim) in enumerate(closest, 1):
        st.write(f"**{rank}.** {w}")
        # La valeur de similarité doit être bornée entre 0.0 et 1.0 pour st.progress
        progress_val = max(0.0, min(1.0, sim))
        st.progress(progress_val)
        st.caption(f"Similarité cosinus : **{sim:.4f}**")
        st.divider()

def compare(word_a, word_b, word_c, top_n=5):
    """Effectue l'opération vectorielle : Vecteur(A) - Vecteur(B) + Vecteur(C)."""
    vec_a = get_word_vector(word_a)
    vec_b = get_word_vector(word_b)
    vec_c = get_word_vector(word_c)

    # Vérification de la présence des mots dans le vocabulaire
    missing = []
    if vec_a is None: missing.append(word_a)
    if vec_b is None: missing.append(word_b)
    if vec_c is None: missing.append(word_c)

    if missing:
        st.error(f"⚠️ Mot(s) absent(s) du vocabulaire : **{', '.join(missing)}**")
        return None

    # Calcul arithmétique sur les vecteurs
    target_vector = vec_a - vec_b + vec_c

    # Recherche des mots les plus proches en excluant les mots de départ
    exclude_words = {word_a.lower().strip(), word_b.lower().strip(), word_c.lower().strip()}
    similarities = []

    for w, idx in tokenizer.word_index.items():
        if idx < vocab_size and w not in exclude_words:
            sim = cosine_similarity(target_vector, embedding_matrix[idx])
            similarities.append((w, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# =========================================================
# 6. WIDGETS : RECHERCHE DE MOTS SIMILAIRES
# =========================================================
st.header("🎯 1. Recherche de mots similaires")

col_input1, col_input2 = st.columns([1, 1])

with col_input1:
    user_word = st.text_input(
        "Entrez un mot en anglais :",
        placeholder="ex : movie, love, terrible, horror..."
    )

with col_input2:
    example_words = ["movie", "film", "good", "bad", "love", "story", "actor", "horror"]
    selected_example = st.selectbox(
        "Ou sélectionnez un mot d'exemple :",
        options=["-- Sélectionner --"] + example_words
    )

# Détermination du mot à chercher
word_to_search = user_word.strip() if user_word.strip() else (selected_example if selected_example != "-- Sélectionner --" else "")

if st.button("🔍 Afficher les 10 mots les plus proches"):
    if word_to_search:
        print_closest(word_to_search, top_n=10)
    else:
        st.info("👆 Veuillez saisir un mot ou en choisir un dans la liste déroulante.")

# =========================================================
# 7. WIDGETS : ARITHMÉTIQUE SÉMANTIQUE (ANALOGIES)
# =========================================================
st.divider()
st.header("🧮 2. Propriétés sémantiques et arithmétiques")
st.write("""
Le modèle Word2Vec préserve les relations sémantiques sous forme géométrique.  
Vous pouvez tester des équations de type : **Mot A - Mot B + Mot C = ?**
""")

col1, col2, col3, col4, col5 = st.columns([3, 1, 3, 1, 3])

with col1:
    word_A = st.text_input("Mot A", value="great")
with col2:
    st.markdown("<h3 style='text-align: center; padding-top: 15px;'>-</h3>", unsafe_allow_html=True)
with col3:
    word_B = st.text_input("Mot B", value="good")
with col4:
    st.markdown("<h3 style='text-align: center; padding-top: 15px;'>+</h3>", unsafe_allow_html=True)
with col5:
    word_C = st.text_input("Mot C", value="bad")

st.caption("💡 **Exemples classiques à tester :**")
st.caption("• *great - good + bad* (résultat attendu : *terrible / awful*)")
st.caption("• *scary - horror + comedy* (résultat attendu : *funny / hilarious*)")
st.caption("• *action - violence + romance* (résultat attendu : *drama / emotional*)")

if st.button("🧮 Calculer l'analogie"):
    if word_A and word_B and word_C:
        results = compare(word_A, word_B, word_C, top_n=5)
        if results:
            st.subheader("🎯 Résultat de l'analogie (Mots les plus proches) :")
            for rank, (w, sim) in enumerate(results, 1):
                col_res, col_val = st.columns([3, 1])
                with col_res:
                    st.write(f"**{rank}. {w}**")
                with col_val:
                    st.write(f"📈 {sim:.4f}")
    else:
        st.warning("⚠️ Veuillez remplir les trois champs de texte.")