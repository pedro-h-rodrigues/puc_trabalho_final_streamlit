import streamlit as st
import pandas as pd
import numpy as np
import spacy
from collections import Counter
import re
import string
import pickle
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer

# Dicionario que possui como chave as categorias (tvs ou veículos)
# E como valores uma lista de tuplas contendo os brigramas mais comuns
dict_path = 'desciptionComponentsDict.plk'
with open(dict_path, 'rb') as file:  
    desciptionComponentsDict = pickle.load(file)

# Modelo de pos tagging
modelo_pos_tag= spacy.load("pt_core_news_sm")


# Pre processamento para limpar a descrição
def pre_process_description(data):
    p1 = re.compile(r'<.*?>')

    # remove html tags
    new_str = p1.sub('', data)

    # remove text inside html text
    p2 = re.compile('(\(\d*\))?(\d)*(... ver número)')
    new_str = p2.sub('', new_str)

    text = ""
    for i in new_str:
        if i in string.punctuation:
            new_str = new_str.replace(i, "")
    
    portugues_stops = stopwords.words('portuguese')
    preprocessed = [
                  w.lower() for w in text.split() 
                  if w not in portugues_stops
                  ]
    
    data = ' '.join(preprocessed)

    #aplica lower
    new_str = new_str.lower()

    marcas = ['lg', 'multilaser', 'philips','xiaomi','sony','sansung','samsung','google','acer','htc','geforce','sapphire','jbl','philips','hp','philco','ryzen','cce','panasonic', 'semp']
    new_str = re.sub('[0-9]{1,}', '[NUMERO]', new_str)

    new_str = re.sub(r'https?:\/\/.*[\r\n]*', ' ', new_str, flags=re.MULTILINE)

    for m in marcas:
        if m in new_str:
            new_str = new_str.replace(m, "[MARCA]")
    
    return new_str

def get_tfidf_top_features(documents,n_top=20):
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, ngram_range =(2,2))
    tfidf = tfidf_vectorizer.fit_transform(documents)
    importance = np.argsort(np.asarray(tfidf.sum(axis=0)).ravel())[::-1]
    tfidf_feature_names = np.array(tfidf_vectorizer.get_feature_names())
    return tfidf_feature_names[importance[:n_top]]

  
def get_score(description, option):
    file = {
        'tvs': 'top_features_tv.csv',
        'carros': 'top_features_carros.csv'
    }
    df_olx = pd.read_csv(file[option])
    res = df_olx[df_olx['Feat'].isin(description)].drop_duplicates(['Feat'])
    res['Tfs'].sum()
    return (res['Tfs'].sum(),res['Feat'].to_list())


def get_recommendation(description,top_bigrams,modelo_spacy):
    '''Recebe como parâmetro a descrição do anúncio,
    Uma lista contendo os bigramas mais comuns para aquela categoria
    e um modelo de pos taging para aplicar na descrição e retorna '''
    description = pre_process_description(description)
    description = modelo_spacy(description)
    
    description_bigrams =[]
    for word in range(1,len(description)):
        description_bigrams.append(description[word-1].text+" "+description[word].text)

    top_bigrams_keys = []
    for tipo in top_bigrams.keys():
        bigrams = top_bigrams[tipo]
        top_bigrams_keys.extend([i[0] for i in bigrams])

    #recommendations = [k for k in top_bigrams_keys if k not in description_bigrams]
    recommendations = []

    #print(description)
    for k in top_bigrams_keys: 
        if k not in description_bigrams: 
            desc = str(description).split()
            if k.split()[0] not in desc and k.split()[1] not in desc:
                recommendations.append(k)
    
    range_recommendations = min(len(recommendations),20)
    
    recommendations = recommendations[:range_recommendations]

    return recommendations


st.title('Recomendador de descrições de sites de anúncios')

option = st.selectbox(
    'Categoria',
    ('tvs', 'carros'))

txt = st.text_area('Descrição para analisar')

result = get_recommendation(txt,desciptionComponentsDict[option],modelo_pos_tag)
if txt:
    score, key_points = get_score(get_tfidf_top_features([pre_process_description(txt)]), option)
    st.write('Pontuação:',score)
st.write('Sugestões:', result)



