# Hybrid Recommender System

## Introduction

This project focuses on building a hybrid recommender system that combines user-based and item-based recommendation approaches for movie suggestions. We utilize movie data and user ratings to provide personalized recommendations to users.

## Data Description

The dataset contains two primary tables:

### Movies Table

- **movieId**: A unique identifier for each movie.
- **title**: The name of the movie.
- **genres**: The type or genre of the movie.

### Ratings Table

- **userId**: A unique identifier for each user.
- **movieId**: A unique identifier for each movie.
- **rating**: The rating given to the movie by the user.
- **timestamp**: The date of the movie evaluation.

## Implementation

This hybrid recommender system is divided into two main components: User-Based Recommendation and Item-Based Recommendation.

### User-Based Recommendation

1. The system selects a random user for recommendation.
2. Movies watched by the selected user are determined, and only common movies are considered.
3. Users who have watched a similar set of movies (at least 60%) to the selected user are identified.
4. Correlation coefficients are computed to find users with similar preferences.
5. The top users with high correlation (above 0.65) are selected.
6. Users' ratings are used to calculate a weighted recommendation score for movies.
7. The top 5 movies with the highest weighted ratings are recommended.

### Item-Based Recommendation

1. The most recently watched and highest-rated movie by the user is selected.
2. Movies with similar user preferences are identified based on the selected movie.
3. Recommendations are made based on the correlation between the selected movie and other movies.
4. The top 5 recommended movies are suggested.

## Usage

You can use this hybrid recommender system by selecting a random user and obtaining recommendations based on their preferences. Additionally, you can provide the name of a movie for which recommendations will be generated. The system utilizes both user behavior and movie correlations to provide accurate and personalized suggestions.

Example:

```python
# Generate recommendations for a random user
random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).iloc[0])
recommendations_user_based = user_based_recommendations(user_movie_df, random_user)

# Generate item-based recommendations for a movie
movie_name = "The Shawshank Redemption"
recommendations_item_based = item_based_recommendations(user_movie_df, movie_name)
```
