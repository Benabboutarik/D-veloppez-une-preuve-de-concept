import streamlit as st
import pandas as pd
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Dashboard de Prédiction")

# Charger le tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Charger la configuration du modèle pré-entraîné
config = RobertaConfig.from_pretrained('roberta-base')

# Créer une instance du modèle avec la configuration
model = RobertaForSequenceClassification(config)

# Charger les poids à partir du fichier local
model_weights_path = 'C:/Users/benab/OneDrive/Documents/OC/OC_P10/roberta_model.pth'
model.load_state_dict(torch.load(model_weights_path))

# Mettre le modèle en mode évaluation
model.eval()

# Charger les données nettoyées et tokenisées depuis le fichier CSV
data_cleaned = pd.read_csv('C:/Users/benab/OneDrive/Documents/OC/OC_P10/data_cleaned.csv')

# Supprimer les lignes avec des valeurs manquantes dans la colonne 'cleaned_text'
data_cleaned = data_cleaned.dropna(subset=['cleaned_text'])

# Créer une liste de commentaires positifs après avoir supprimé les valeurs manquantes
positive_comments = data_cleaned[data_cleaned['label'] == 'positif']['cleaned_text'].tolist()

# Créer une liste de commentaires positifs
positive_comments = data_cleaned[data_cleaned['label'] == 'positif']['cleaned_text'].tolist()

# Créer un objet CountVectorizer pour les commentaires positifs
positive_count_vectorizer = CountVectorizer()
positive_X_counts = positive_count_vectorizer.fit_transform(positive_comments)
positive_feature_names = positive_count_vectorizer.get_feature_names_out()
positive_word_frequencies = positive_X_counts.sum(axis=0).A1

# Créer un dictionnaire de fréquence des mots pour les commentaires positifs
positive_word_frequencies_dict = dict(zip(positive_feature_names, positive_word_frequencies))

# Trier le dictionnaire par fréquence des mots
sorted_positive_word_frequencies = dict(sorted(positive_word_frequencies_dict.items(), key=lambda x: x[1], reverse=True))

# Sélectionner les 20 premiers mots
top_20_positive_words = dict(list(sorted_positive_word_frequencies.items())[:20])

# Créer une liste de commentaires négatifs
negative_comments = data_cleaned[data_cleaned['label'] == 'négatif']['cleaned_text'].tolist()

# Créer un objet CountVectorizer pour les commentaires négatifs
negative_count_vectorizer = CountVectorizer()
negative_X_counts = negative_count_vectorizer.fit_transform(negative_comments)
negative_feature_names = negative_count_vectorizer.get_feature_names_out()
negative_word_frequencies = negative_X_counts.sum(axis=0).A1

# Créer un dictionnaire de fréquence des mots pour les commentaires négatifs
negative_word_frequencies_dict = dict(zip(negative_feature_names, negative_word_frequencies))

# Trier le dictionnaire par fréquence des mots
sorted_negative_word_frequencies = dict(sorted(negative_word_frequencies_dict.items(), key=lambda x: x[1], reverse=True))

# Sélectionner les 20 premiers mots
top_20_negative_words = dict(list(sorted_negative_word_frequencies.items())[:20])

# Afficher les premières lignes du dataset dans un tableau
st.subheader("Les Premières Lignes du Dataset :")
st.write(data_cleaned.head())

#  la destribition des labels sur le dataset train 
st.subheader("Distribution des labels")
plt.figure(figsize=(6, 4))
data_cleaned['label'].value_counts().plot(kind='bar', color=['red', 'green'])
plt.title('Distribution des labels')
plt.xlabel('Label')
plt.ylabel("Nombre d'occurrences")
plt.xticks(rotation='horizontal')
st.pyplot()

# Ajouter une section pour les 20 premiers mots par fréquence
st.subheader("Fréquence des 20 premiers mots")

# Calculez les 20 premiers mots par fréquence
count_vectorizer = CountVectorizer()
X_counts = count_vectorizer.fit_transform(data_cleaned['cleaned_text'])
feature_names = count_vectorizer.get_feature_names_out()
word_frequencies = X_counts.sum(axis=0).A1
word_frequencies_dict = dict(zip(feature_names, word_frequencies))
sorted_word_frequencies = dict(sorted(word_frequencies_dict.items(), key=lambda x: x[1], reverse=True))
top_20_words = dict(list(sorted_word_frequencies.items())[:20])

# Afficher le graphique
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(top_20_words.keys(), top_20_words.values())
ax.set_xlabel('Mots')
ax.set_ylabel('Fréquence')
ax.set_title('Fréquence des 20 premiers mots')
ax.tick_params(axis='x', rotation=45)
st.pyplot(fig)

# Génération du nuage de mots pour les commentaires positifs
st.subheader("Nuage de mots pour les commentaires positifs")
positive_comments = data_cleaned[data_cleaned['label'] == 'positif']['cleaned_text'].str.cat(sep=' ')
wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(positive_comments)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.axis('off')
st.pyplot(plt)


# Afficher le graphique des 20 premiers mots pour les commentaires positifs
st.subheader("Fréquence des 20 premiers mots pour les commentaires positifs :")
plt.figure(figsize=(12, 6))
plt.bar(top_20_positive_words.keys(), top_20_positive_words.values())
plt.xlabel('Mots')
plt.ylabel('Fréquence')
plt.title('Fréquence des 20 premiers mots pour les commentaires positifs')
plt.xticks(rotation='vertical')
st.pyplot()

# Génération du nuage de mots pour les commentaires négatifs
st.subheader("Nuage de mots pour les commentaires négatifs")
negative_comments = data_cleaned[data_cleaned['label'] == 'négatif']['cleaned_text'].str.cat(sep=' ')
wordcloud_negative = WordCloud(width=800, height=400, background_color='white').generate(negative_comments)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.axis('off')
st.pyplot(plt)

# Afficher le graphique des 20 premiers mots pour les commentaires négatifs
st.subheader("Fréquence des 20 premiers mots pour les commentaires négatifs :")
plt.figure(figsize=(12, 6))
plt.bar(top_20_negative_words.keys(), top_20_negative_words.values())
plt.xlabel('Mots')
plt.ylabel('Fréquence')
plt.title('Fréquence des 20 premiers mots pour les commentaires négatifs')
plt.xticks(rotation='vertical')
st.pyplot()


# Ajouter une section pour l'histogramme des longueurs de texte
st.subheader("Distribution des longueurs de texte")

# Calculez les longueurs des textes dans le jeu de données
text_lengths = [len(text.split()) for text in data_cleaned['cleaned_text']]

# Affichez un histogramme des longueurs de texte
fig, ax = plt.subplots()
ax.hist(text_lengths, bins=30)
ax.set_xlabel('Longueur du texte')
ax.set_ylabel('Nombre de textes')
ax.set_title('Distribution des longueurs de texte')
st.pyplot(fig)

df_performance = pd.read_csv('C:/Users/benab/OneDrive/Documents/OC/OC_P10/df_performance.csv')

st.title('Performance des modèles')
st.write(df_performance.head())


# Interface utilisateur Streamlit pour la prédiction
st.title("Prédiction de Sentiment")

# Champ de saisie de texte pour entrer le texte à prédire
text_input = st.text_input("Entrez le texte à prédire :", "")

# Bouton pour déclencher la prédiction
if st.button("Prédire"):
    try:
        # Convertir le texte en tensor avec le tokenizer de RoBERTa
        inputs = tokenizer(text_input, return_tensors="pt")

        # Faire la prédiction avec le modèle RoBERTa
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            predicted_label_roberta = torch.argmax(probabilities, dim=1).item()

        # Afficher la prédiction
        if predicted_label_roberta == 1:
            st.write("Prédiction: Le texte est Positif.")
        else:
            st.write("Prédiction: Le texte est Négatif.")
    except Exception as e:
        st.write(f"Erreur lors de la prédiction : {str(e)}")