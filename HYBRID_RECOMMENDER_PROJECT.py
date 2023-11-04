#############################################
# Hybrid Recommender System
#############################################

# import random
import pandas as pd

pd.set_option("display.max_columns", None)

movie = pd.read_csv("datasets/movie.csv")
rating = pd.read_csv("datasets/rating.csv")

# movie.head()
# movie.columns
# rating.head()
# rating.columns

df = movie.merge(rating, how="left", on="movieId")
df.head()

comment_counts = pd.DataFrame(df["title"].value_counts())
rare_movies = comment_counts[comment_counts["count"] < 1000].index
common_movies = df[~df["title"].isin(rare_movies)]

user_movie_df = common_movies.pivot_table(
    index=["userId"], columns=["title"], values="rating"
)
user_movie_df.head()


#############################################
# Determining the Movies Watched by the User for Recommendation
#############################################

random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).iloc[0])
# random_user = random.choice(df["userId"])

random_user_df = user_movie_df[user_movie_df.index == random_user]
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

#############################################
# Accessing Data and Ids of Other Users Watching the Same Movies
#############################################

# Select the columns of the movies watched by the selected user from user_movie_df and create a new dataframe named movies_watched_df.
movies_watched_df = user_movie_df[movies_watched]
# movies_watched_df.shape

user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userID", "movie_count"]

# user_movie_count.columns

# Step 3: We consider users who have watched 60 percent or more of the movies voted by the selected user as similar users.
# Create a list called users_same_movies from the ids of these users.
perc = len(movies_watched) * 60 / 100
user_same_movies = user_movie_count[user_movie_count["movie_count"] >= perc]["userID"]

#############################################
# Identifying the Users Most Similar to the User for Recommendation
#############################################

# Create a new corr_df dataframe where the correlations of the users with each other will be found.
final_df = pd.concat(
    [
        movies_watched_df[movies_watched_df.index.isin(user_same_movies)],
        random_user_df[movies_watched],
    ]
)


final_df = final_df.drop_duplicates()
corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ["user_id_1", "user_id_2"]
corr_df = corr_df.reset_index()

corr_df[corr_df["user_id_1"] == random_user]


# Create a new dataframe called top_users by filtering the users with high correlation (above 0.65) with the selected user.
top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][
    ["user_id_2", "corr"]
].reset_index(drop=True)


top_users = top_users.sort_values(by="corr", ascending=False)
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

rating = pd.read_csv("datasets/rating.csv")
top_users_ratings = top_users.merge(
    rating[["userId", "movieId", "rating"]], how="inner"
)

top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]

#############################################
# Calculating the Weighted Average Recommendation Score and Keeping the Top 5 Movies
#############################################

top_users_ratings["weighted_rating"] = (
    top_users_ratings["corr"] * top_users_ratings["rating"]
)

recommendation_df = top_users_ratings.groupby("movieId").agg(
    {"weighted_rating": "mean"}
)
recommendation_df = recommendation_df.reset_index()

# In recommendation_df, select movies with a weighted rating greater than 3.5 and sort by weighted rating.
recommendation_df[recommendation_df["weighted_rating"] > 3.5]

movies_to_be_recommend = (
    recommendation_df[recommendation_df["weighted_rating"] > 3.5]
    .sort_values("weighted_rating", ascending=False)
    .head(5)
)


movie = pd.read_csv("datasets/movie.csv")
movies_to_be_recommend = movies_to_be_recommend.merge(movie[["movieId", "title"]])

#############################################
# Item-Based Recommendation
#############################################

# Make item-based suggestions based on the name of the most recently watched and highest rated movie by the user.
user = 108170

movie = pd.read_csv("datasets/movie.csv")
rating = pd.read_csv("datasets/rating.csv")
df = movie.merge(rating, how="left", on="movieId")

# Adım 2: Get the id of the movie with the most recent score among the movies that the user to be recommended has given 5 points.
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["timestamp"].max()
user_df = df[df["userId"] == user]
user_df_5 = (
    user_df[user_df["rating"] == 5.0]
    .sort_values(by="timestamp", ascending=False)
    .head(5)
)
top_1_movie = user_df_5["title"].iloc[0]
movie_name = user_movie_df[top_1_movie]

# Filter the user_movie_df dataframe created in the user based recommendation section according to the selected movie id.
filtered_movies = user_movie_df[user_movie_df.index == top_1_movie]

# Using the filtered dataframe, find the correlation between the selected movie and other movies and sort them.

recomm = (
    user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(6).iloc[1:]
)
