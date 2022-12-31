import pickle

import pandas as pd
import streamlit as st
import requests

from sklearn.feature_extraction.text import TfidfVectorizer




def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(
        movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path

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
    print(recommended_movie_names)
    return recommended_movie_names, recommended_movie_posters, distances





# Trending MOvies
#*********************************************************************************************************************
st.markdown('<h1 style="color: purple;">Movies Recommender System</h1>', unsafe_allow_html=True)

st.markdown('<h3 style="color: red;">Trending Movies</h3>', unsafe_allow_html=True)

T_df = pickle.load(open('model/data_dict.pkl', 'rb'))
df = pd.DataFrame(T_df)
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


def cosine_sim():
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









# ALGORITHM 2
#************************************************************************************************************************

algo2 = pickle.load(open('model/algo2data.pkl', 'rb'))

algo2data = pd.DataFrame(algo2)



plot = algo2data['overview'].tolist()

cast = algo2data['cast'].tolist()

movie_descriptions = [' '.join(str(t) for t in s) for s in zip(plot, cast)]

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(movie_descriptions)

# calculate the cosine similarity between all pairs of movies

from sklearn.metrics.pairwise import cosine_similarity
similarityalgo2=cosine_similarity(X,X)

# create a mapping of movie titles to indices
indices = pd.Series(algo2data.index, index=algo2data['title']).drop_duplicates()



def recommend_movies(title, n=10, similarity=similarityalgo2):
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





































if st.button('Show Recommendation'):
    if selected_algo == 'Cosine Similarity':
        cosine_sim()
    elif selected_algo == 'Content_Based TFID_VECTORIZATION':
        algo2()
    # elif selected_algo == 'Hybrid Approach':
    #     st.image()
    # else:
    #     st.text("NO RESULT")
