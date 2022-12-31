import pickle

import pandas as pd
import streamlit as st
import requests

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(
        movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    print(full_path)
    return full_path


def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:12]:
        # fetch the movie poster
        movie_id = movies.iloc[i[0]].id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)

    return recommended_movie_names, recommended_movie_posters, distances




# Trending MOvies

st.markdown('<h1 style="color: purple;">Movies Recommender System</h1>', unsafe_allow_html=True)

st.markdown('<h3 style="color: red;">Trending Movies</h3>', unsafe_allow_html=True)

T_df = pickle.load(open('model/data_dict.pkl', 'rb'))
#df = pd.DataFrame(T_df)
data = pd.DataFrame(T_df)

C = data['vote_average'].mean()
print(C)

m = data['vote_count'].quantile(0.9)
print(m)


def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v / (v + m) * R) + (m / (m + v) * C)


data['score'] = data.apply(weighted_rating, axis=1)
data = data.sort_values('score', ascending=False)
# Print the top 10 movies

Tlist = data[['id', 'title']].head(10)
trendlist = list(data['id'].head(10))
print("Trending List", trendlist)
recommended_movie_posters = []
recommended_movie_names = []

for i in data['title'].head(10):
    recommended_movie_names.append(i)

for i in trendlist:
    recommended_movie_posters.append(fetch_poster(i))

col11, col12, col13 = st.columns(3)
col14, col15, col16 = st.columns(3)
col17, col18, col19 = st.columns(3)

with col11:
    st.text(recommended_movie_names[0])
    st.image(recommended_movie_posters[0])
with col12:
    st.text(recommended_movie_names[1])
    st.image(recommended_movie_posters[1])
with col13:
    st.text(recommended_movie_names[2])
    st.image(recommended_movie_posters[2])
with col14:
    st.text(recommended_movie_names[3])
    st.image(recommended_movie_posters[3])
with col15:
    st.text(recommended_movie_names[4])
    st.image(recommended_movie_posters[4])
with col16:
    st.text(recommended_movie_names[5])
    st.image(recommended_movie_posters[5])
with col17:
    st.text(recommended_movie_names[6])
    st.image(recommended_movie_posters[6])
with col18:
    st.text(recommended_movie_names[7])
    st.image(recommended_movie_posters[7])
with col19:
    st.text(recommended_movie_names[8])
    st.image(recommended_movie_posters[8])

st.markdown('<h3 style="color: red;">Try Recommendation System</h3>', unsafe_allow_html=True)

movie_dict = pickle.load(open('model/movies_dict.pkl', 'rb'))

movies = pd.DataFrame(movie_dict)

similarity = pickle.load(open('model/similarity.pkl', 'rb'))

algo_list = ['Cosine Similarity', 'Content_Based TFID_VECTORIZATION', 'Hybrid Approach']
movie_list = movies['title'].values
selected_movie = st.selectbox("Type or select a movie from the dropdown", movie_list)
selected_algo = st.selectbox("Select a Recommendation Algorithm", algo_list)



# ALGORITHM 2


algo2 = pickle.load(open('model/algo2data.pkl', 'rb'))

algo2data = pd.DataFrame(algo2)



plot = algo2data['overview'].tolist()

cast = algo2data['cast'].tolist()

movie_descriptions = [' '.join(str(t) for t in s) for s in zip(plot, cast)]

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(movie_descriptions)

# calculate the cosine similarity between all pairs of movies
similarity2 = cosine_similarity(X,X)

# create a mapping of movie titles to indices
indices = pd.Series(algo2data.index, index=algo2data['title']).drop_duplicates()



def recommend_movies(title, n=10, similarity=similarity2):
    # get the index of the movie
    idx = indices[title]

    sim_scores = list(enumerate(similarity[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:n + 1]

    movie_indices = [i[0] for i in sim_scores]

    mnames = []
    mids = []
    for i in movie_indices:
        mnames.append(algo2data['title'].iloc[i])
    for i in movie_indices:
        mids.append(algo2data['id'].iloc[i])

    print(mnames)
    print(mids)
    return mnames,mids


# test the recommendation function
# recommend_movies('Avatar')



def algo2():
    mnames,mid= recommend_movies(selected_movie)
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    col7, col8, col9 = st.columns(3)

    with col1:
        st.text(mnames[0])
        st.image(fetch_poster(mid[0]))

    with col2:
        st.text(mnames[1])
        st.image(fetch_poster(mid[1]))

    with col3:
        st.text(mnames[3])
        st.image(fetch_poster(mid[3]))

    with col4:
        st.text(mnames[4])
        st.image(fetch_poster(mid[4]))
    with col5:
        st.text(mnames[5])
        st.image(fetch_poster(mid[5]))

    with col6:
        st.text(mnames[6])
        st.image(fetch_poster(mid[6]))

    with col7:
        st.text(mnames[7])
        st.image(fetch_poster(mid[7]))
    with col8:
        st.text(mnames[8])
        st.image(fetch_poster(mid[8]))

    with col9:
        st.text(mnames[9])
        st.image(fetch_poster(mid[9]))



#ALGO 3 HYBRID


algo3 = pickle.load(open('model/algo3data.pkl', 'rb'))

df = pd.DataFrame(algo3)


R = df['vote_average']
v =df['vote_count']
# We will only consider movies that have more votes than at least 80% of the movies in our dataset
m = df['vote_count'].quantile(0.8)
C = df['vote_average'].mean()
df['weighted_average'] = (R*v + C*m)/(v+m)



from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[['popularity', 'weighted_average']])
weighted_df = pd.DataFrame(scaled, columns=['popularity', 'weighted_average'])

weighted_df.index = df['original_title']

weighted_df['score'] = weighted_df['weighted_average']*0.4 + weighted_df['popularity'].astype('float64')*0.6

weighted_df_sorted = weighted_df.sort_values(by='score', ascending=False)
#weighted_df_sorted.head(10)
hybrid_df = df[['id','original_title', 'adult', 'genres', 'overview', 'production_companies', 'tagline', 'keywords', 'crew', 'characters', 'actors']]




import string
import re
def separate(text):
    clean_text = []
    for t in text.split(','):
        cleaned = re.sub('\(.*\)', '', t) # Remove text inside parentheses
        cleaned = cleaned.translate(str.maketrans('','', string.digits))
        cleaned = cleaned.replace(' ', '')
        cleaned = cleaned.translate(str.maketrans('','', string.punctuation)).lower()
        clean_text.append(cleaned)
    return ' '.join(clean_text)

def remove_punc(text):
    cleaned = text.translate(str.maketrans('','', string.punctuation)).lower()
    clean_text = cleaned.translate(str.maketrans('','', string.digits))
    return clean_text



hybrid_df['adult'] = hybrid_df['adult'].apply(remove_punc)
hybrid_df['genres'] = hybrid_df['genres'].apply(remove_punc)
hybrid_df['overview'] = hybrid_df['overview'].apply(remove_punc)
hybrid_df['production_companies'] = hybrid_df['production_companies'].apply(separate)
hybrid_df['tagline'] = hybrid_df['tagline'].apply(remove_punc)
hybrid_df['keywords'] = hybrid_df['keywords'].apply(separate)
hybrid_df['crew'] = hybrid_df['crew'].apply(separate)
hybrid_df['characters'] = hybrid_df['characters'].apply(separate)
hybrid_df['actors'] = hybrid_df['actors'].apply(separate)

hybrid_df['bag_of_words'] = ''
hybrid_df['bag_of_words'] = hybrid_df[hybrid_df.columns[1:]].apply(lambda x: ' '.join(x), axis=1)
hybrid_df.set_index('original_title', inplace=True)

hybrid_df = hybrid_df[['id','bag_of_words']]
hybrid_df.head()


hybrid_df = weighted_df_sorted[:10000].merge(hybrid_df, left_index=True, right_index=True, how='left')

tfidf = TfidfVectorizer(stop_words='english', min_df=5)
tfidf_matrix = tfidf.fit_transform(hybrid_df['bag_of_words'])
tfidf_matrix.shape

cos_sim = cosine_similarity(tfidf_matrix)
cos_sim.shape


def predict(title, similarity_weight=0.7, top_n=10):
    data = hybrid_df.reset_index()

    index_movie = data[data['original_title'] == title].index
    similarity = cos_sim[index_movie].T

    sim_df = pd.DataFrame(similarity, columns=['similarity'])
    final_df = pd.concat([data, sim_df], axis=1)
    # print(final_df.head(5))

    # You can also play around with the number
    final_df['final_score'] = final_df['score'] * (1 - similarity_weight) + final_df['similarity'] * similarity_weight

    final_df_sorted = final_df.sort_values(by='final_score', ascending=False).head(top_n)
    # final_df_sorted.set_index('original_title', inplace=True)
    # final_df_sorted[['id','score', 'similarity', 'final_score']]

    liname = []
    for i in final_df_sorted['original_title']:
        liname.append(i)
    print(liname)

    liID = []
    for i in final_df_sorted['id']:
        liID.append(i)
    print(liID)

    return liname, liID



def algo3system():
    liname, liID= predict(selected_movie, similarity_weight=0.7, top_n=10)
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    col7, col8, col9 = st.columns(3)

    with col1:
        st.text(liname[1])
        st.image(fetch_poster(liID[1]))

    with col2:
        st.text(liname[2])
        st.image(fetch_poster(liID[2]))

    with col3:
        st.text(liname[3])
        st.image(fetch_poster(liID[3]))

    with col4:
        st.text(liname[4])
        st.image(fetch_poster(liID[4]))
    with col5:
        st.text(liname[5])
        st.image(fetch_poster(liID[5]))

    with col6:
        st.text(liname[6])
        st.image(fetch_poster(liID[6]))

    with col7:
        st.text(liname[7])
        st.image(fetch_poster(liID[7]))
    with col8:
        st.text(liname[8])
        st.image(fetch_poster(liID[8]))

    with col9:
        st.text(liname[9])
        st.image(fetch_poster(liID[9]))


def cosine_similarity():
    recommended_movie_names, recommended_movie_posters, distances = recommend(selected_movie)
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    col7, col8, col9 = st.columns(3)

    with col1:
        st.text(recommended_movie_names[0])
        st.image(recommended_movie_posters[0])

    with col2:
        st.text(recommended_movie_names[1])
        st.image(recommended_movie_posters[1])

    with col3:
        st.text(recommended_movie_names[2])
        st.image(recommended_movie_posters[2])
    with col4:
        st.text(recommended_movie_names[3])
        st.image(recommended_movie_posters[3])
    with col5:
        st.text(recommended_movie_names[4])
        st.image(recommended_movie_posters[4])

    with col6:
        st.text(recommended_movie_names[5])
        st.image(recommended_movie_posters[5])
    with col7:
        st.text(recommended_movie_names[6])
        st.image(recommended_movie_posters[6])

    with col8:
        st.text(recommended_movie_names[7])
        st.image(recommended_movie_posters[7])
    with col9:
        st.text(recommended_movie_names[8])
        st.image(recommended_movie_posters[8])














if st.button('Show Recommendation'):
    if selected_algo == 'Cosine Similarity':
        cosine_similarity()
    elif selected_algo == 'Content_Based TFID_VECTORIZATION':
        algo2()
    elif selected_algo == 'Hybrid Approach':
        algo3system()
    # else:
    #     st.text("NO RESULT")
