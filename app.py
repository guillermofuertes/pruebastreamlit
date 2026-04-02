import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.title("Demo de Embeddings en vivo")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Inputs
word1 = st.text_input("Palabra 1", "gato")
word2 = st.text_input("Palabra 2", "perro")

if st.button("Calcular similitud"):
    emb1 = model.encode([word1])[0]
    emb2 = model.encode([word2])[0]

    sim = cosine_similarity([emb1], [emb2])[0][0]

    st.subheader("Primeros valores del embedding")
    st.write([round(x, 3) for x in emb1[:10]])

    st.subheader("Similitud")
    st.metric("Similitud coseno", f"{sim:.3f}")

    # Interpretación automática
    if sim > 0.7:
        st.success("Muy relacionados")
    elif sim > 0.4:
        st.warning("Algo relacionados")
    else:
        st.error("Poco relacionados")
      
print("\nSimilitudes:")
for i in range(len(words)):
    for j in range(i+1, len(words)):
        sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
        print(f"{words[i]} vs {words[j]} = {sim:.3f}")

# 7. Visualización simple
for i, word in enumerate(words):
    plt.plot(embeddings[i], label=word)

plt.legend()
plt.title("Embeddings como señales numéricas")
plt.show()
