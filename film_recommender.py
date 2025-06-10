import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm, trange
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

# Set device to CUDA if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# Additional CUDA optimizations for better performance
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
    torch.backends.cuda.matmul.allow_tf32 = True  # Use TensorFloat-32 for faster matmul
    torch.backends.cudnn.allow_tf32 = True  # Use TensorFloat-32 for convolutions

DATASET_FOLDER = "dataset"
RATINGS_FILE = os.path.join(DATASET_FOLDER, "ratings.csv")
MOVIES_FILE = os.path.join(DATASET_FOLDER, "movies.csv")

df_ratings = pd.read_csv(RATINGS_FILE)
df_movies = pd.read_csv(MOVIES_FILE)

# Prepare lists for parallel processing
movieId_list = df_movies.movieId.tolist()
title_list = df_movies.title.tolist()
genre_list = df_movies.genres.tolist()

# clean the ratings data
df_ratings = df_ratings.dropna()
df_ratings["movieId"] = df_ratings["movieId"].astype(int, copy=False)

# let's only work with movies with enough ratings.
min_ratings_per_movie = 1000

# get the number of ratings per movie
df_movies_to_num_ratings = df_ratings.groupby("movieId", as_index=False)[
    "rating"
].count()
print("total movies in corpus: ", len(df_movies_to_num_ratings))

df_movies_to_num_ratings = df_movies_to_num_ratings.sort_values(
    by=["rating"], ascending=False
)  # type: ignore
df_movies_to_num_ratings = df_movies_to_num_ratings[
    df_movies_to_num_ratings["rating"] > min_ratings_per_movie
]
print("movies with enough ratings: ", len(df_movies_to_num_ratings))

# get list of the top movies by number of ratings.
top_movies = df_movies_to_num_ratings.movieId.tolist()

# Now create rating_list for parallel processing
rating_list = df_movies_to_num_ratings.rating.tolist()

# Only now, after df_ratings_final is defined, do the parallel processing

df_ratings_final = df_ratings[df_ratings.movieId.isin(top_movies)]

from concurrent.futures import ThreadPoolExecutor

movieId_to_num_ratings = {}

# Use filtered lists for parallel processing
filtered_movieId_list = df_movies_to_num_ratings.movieId.tolist()
filtered_rating_list = df_movies_to_num_ratings.rating.tolist()


def build_movieId_to_num_ratings(i, movieId_list, rating_list):
    return (movieId_list[i], rating_list[i])


with ThreadPoolExecutor() as executor:
    for movieId, rating in executor.map(
        build_movieId_to_num_ratings,
        range(len(filtered_movieId_list)),
        [filtered_movieId_list] * len(filtered_movieId_list),
        [filtered_rating_list] * len(filtered_movieId_list),
    ):
        movieId_to_num_ratings[movieId] = rating

movieId_to_title = {}
title_to_movieId = {}


def build_title_maps(i, movieId_list, title_list):
    movieId = movieId_list[i]
    title = title_list[i]
    return (movieId, title)


with ThreadPoolExecutor() as executor:
    for movieId, title in executor.map(
        build_title_maps,
        range(len(movieId_list)),
        [movieId_list] * len(movieId_list),
        [title_list] * len(movieId_list),
    ):
        movieId_to_title[movieId] = title
        title_to_movieId[title] = movieId

genres = set()
movieId_to_genres = {}


def build_genre_maps(i, movieId_list, genre_list, top_movies):
    movieId = movieId_list[i]
    if movieId not in top_movies:
        return None
    genre_set = set()
    for genre in genre_list[i].split("|"):
        genre_set.add(genre)
    return (movieId, genre_set, genre_list[i])


with ThreadPoolExecutor() as executor:
    for result in executor.map(
        build_genre_maps,
        range(len(movieId_list)),
        [movieId_list] * len(movieId_list),
        [genre_list] * len(movieId_list),
        [top_movies] * len(movieId_list),
    ):
        if result is not None:
            movieId, genre_set, genre_str = result
            movieId_to_genres[movieId] = genre_set
            for genre in genre_set:
                genres.add(genre)

df_movies_to_avg_rating = df_ratings_final.groupby("movieId", as_index=False)[
    "rating"
].mean()

movieId_to_avg_rating = {}

movieId_list = df_movies_to_avg_rating.movieId.tolist()
rating_list = df_movies_to_avg_rating.rating.tolist()
for i in range(len(movieId_list)):
    movieId_to_avg_rating[movieId_list[i]] = rating_list[i]
item_emb_movieId_to_i = {s: i for i, s in enumerate(top_movies)}
item_emb_i_to_movieId = {i: s for s, i in item_emb_movieId_to_i.items()}

# build ITEM genre feature context
genre_to_i = {s: i for i, s in enumerate(genres)}
i_to_genre = {i: s for s, i in genre_to_i.items()}
num_movies_for_user_context = 250
user_context_movies = top_movies[:num_movies_for_user_context]
# aggregate dataframe down into one row per user and list of their movies and ratings.
df_ratings_aggregated = (
    df_ratings_final.groupby("userId")
    .agg({"movieId": lambda x: list(x), "rating": lambda y: list(y)})
    .reset_index()
)
# build the USER context
user_context_size = len(user_context_movies) + len(genres)
# for every movie, create a training example feature context vector lookup
# it will contain the movie's genres.
movieId_to_context = {}
for movieId in top_movies:
    context = [0.0] * len(genres)

    for genre in movieId_to_genres[movieId]:
        context[genre_to_i[genre]] = float(1.0)

    movieId_to_context[movieId] = context


def preprocess_dataset(df_ratings_aggregated):
    user_context_movieId_to_i = {s: i for i, s in enumerate(list(user_context_movies))}
    user_context_i_to_movieId = {i: s for s, i in user_context_movieId_to_i.items()}

    user_context_genre_to_i = {
        s: i + len(user_context_movies) for i, s in enumerate(list(genres))
    }
    user_context_i_to_genre = {i: s for s, i in user_context_genre_to_i.items()}
    percent_ratings_as_watch_history = 0.8

    user_list = df_ratings_aggregated["userId"].tolist()
    movieId_list_list = df_ratings_aggregated["movieId"].tolist()
    rating_list_list = df_ratings_aggregated["rating"].tolist()

    def process_user(i):
        userId = user_list[i]
        movieId_list = movieId_list_list[i]
        rating_list = rating_list_list[i]
        num_rated_movies = len(movieId_list)
        if num_rated_movies <= 5:
            return None
        user_watch = {}
        user_label = {}
        rated_movies = list(zip(movieId_list, rating_list))
        random.shuffle(rated_movies)
        for movieId, rating in rated_movies[
            : int(num_rated_movies * percent_ratings_as_watch_history)
        ]:
            user_watch[movieId] = rating
        for movieId, rating in rated_movies[
            int(num_rated_movies * percent_ratings_as_watch_history) :
        ]:
            user_label[movieId] = rating
        return (userId, user_watch, user_label)

    user_to_movie_to_rating_WATCH_HISTORY = {}
    user_to_movie_to_rating_LABEL = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_user, range(len(user_list))))
    for res in results:
        if res is not None:
            userId, user_watch, user_label = res
            user_to_movie_to_rating_WATCH_HISTORY[userId] = user_watch
            user_to_movie_to_rating_LABEL[userId] = user_label

    user_to_avg_rating = {}
    for user in user_to_movie_to_rating_WATCH_HISTORY.keys():
        user_to_avg_rating[user] = 0
        for movieId in user_to_movie_to_rating_WATCH_HISTORY[user].keys():
            user_to_avg_rating[user] += user_to_movie_to_rating_WATCH_HISTORY[user][
                movieId
            ]
        user_to_avg_rating[user] /= len(
            user_to_movie_to_rating_WATCH_HISTORY[user].keys()
        )

    user_to_genre_to_stat = {}

    def process_user_genre(user):
        genre_stat = {}
        for movieId in user_to_movie_to_rating_WATCH_HISTORY[user].keys():
            for genre in movieId_to_genres[movieId]:
                if genre not in genre_stat:
                    genre_stat[genre] = {"NUM_RATINGS": 0, "SUM_RATINGS": 0}
                genre_stat[genre]["NUM_RATINGS"] += 1
                genre_stat[genre][
                    "SUM_RATINGS"
                ] += user_to_movie_to_rating_WATCH_HISTORY[user][movieId]
        return (user, genre_stat)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        genre_results = list(
            executor.map(
                process_user_genre, user_to_movie_to_rating_WATCH_HISTORY.keys()
            )
        )
    for user, genre_stat in genre_results:
        user_to_genre_to_stat[user] = genre_stat
    for user in user_to_genre_to_stat.keys():
        for genre in user_to_genre_to_stat[user].keys():
            num_ratings = user_to_genre_to_stat[user][genre]["NUM_RATINGS"]
            sum_ratings = user_to_genre_to_stat[user][genre]["SUM_RATINGS"]
            user_to_genre_to_stat[user][genre]["AVG_RATING"] = sum_ratings / num_ratings
    user_to_context = {}

    def process_user_context(user):
        context = [0.0] * user_context_size
        for movieId in user_to_movie_to_rating_WATCH_HISTORY[user].keys():
            if movieId in user_context_movies:
                context[user_context_movieId_to_i[movieId]] = float(
                    user_to_movie_to_rating_WATCH_HISTORY[user][movieId]
                    - user_to_avg_rating[user]
                )
        for genre in user_to_genre_to_stat[user].keys():
            context[user_context_genre_to_i[genre]] = float(
                user_to_genre_to_stat[user][genre]["AVG_RATING"]
                - user_to_avg_rating[user]
            )
        return (user, context)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        context_results = list(
            executor.map(
                process_user_context, user_to_movie_to_rating_WATCH_HISTORY.keys()
            )
        )
    for user, context in context_results:
        user_to_context[user] = context
    return (user_to_movie_to_rating_LABEL, user_to_context, user_to_avg_rating)


# Build the final Dataset
def build_dataset(
    users,
    user_to_movie_to_rating_LABEL,
    user_to_context,
    item_emb_movieId_to_i,
    movieId_to_context,
    user_to_avg_rating,
):
    # the user context (i.e. the watch hisotyr and genre affinities)
    X = []

    # the movieID for the movie we will predict rating for.
    # used to lookup the movie embedding to feed into the NN item tower.
    target_movieId = []

    # the feature context of the movie we will predict the rating for.
    # will also feed into it's own embedding and will be stacked with the embedding above.
    target_movieId_context = []

    # the predicted rating
    Y = []

    # create training examples, one for each movie the user has that we want as a label.
    for user in users:
        for movieId in user_to_movie_to_rating_LABEL[user].keys():
            X.append(user_to_context[user])

            target_movieId.append(item_emb_movieId_to_i[movieId])

            target_movieId_context.append(movieId_to_context[movieId])

            # remember to debias the user rating so we can learn to predict if user
            # like/dislike a movie based on their features and the movie features.
            Y.append(
                float(
                    user_to_movie_to_rating_LABEL[user][movieId]
                    - user_to_avg_rating[user]
                )
            )

    X = torch.tensor(X, device=DEVICE)
    Y = torch.tensor(Y, device=DEVICE)
    target_movieId = torch.tensor(target_movieId, device=DEVICE)
    target_movieId_context = torch.tensor(target_movieId_context, device=DEVICE)

    return X, Y, target_movieId, target_movieId_context


def train_test_split(
    user_to_movie_to_rating_LABEL,
    user_to_context,
    item_emb_movieId_to_i,
    movieId_to_context,
    user_to_avg_rating,
):

    # user users with enough ratings to predict to be useful for model learning.
    final_users = []

    for user in user_to_movie_to_rating_LABEL.keys():
        num_ratings = len(user_to_movie_to_rating_LABEL[user])

        if num_ratings >= 2 and num_ratings < 500:
            final_users.append(user)
    # split users into train and validation users
    percent_users_train = 0.8

    random.shuffle(final_users)

    train_users = final_users[: int(len(final_users) * percent_users_train)]
    validation_users = final_users[int(len(final_users) * percent_users_train) :]
    X_train, Y_train, target_movieId_train, target_movieId_context_train = (
        build_dataset(
            train_users,
            user_to_movie_to_rating_LABEL,
            user_to_context,
            item_emb_movieId_to_i,
            movieId_to_context,
            user_to_avg_rating,
        )
    )
    X_val, Y_val, target_movieId_val, target_movieId_context_val = build_dataset(
        validation_users,
        user_to_movie_to_rating_LABEL,
        user_to_context,
        item_emb_movieId_to_i,
        movieId_to_context,
        user_to_avg_rating,
    )
    return (
        X_train,
        Y_train,
        target_movieId_train,
        target_movieId_context_train,
        X_val,
        Y_val,
        target_movieId_val,
        target_movieId_context_val,
    )


item_feature_embedding_size = 25
item_movieId_embedding_size = 25

# USER feature tower
user_feature_embedding_size = (
    50  # must be the concat dimension of both item embeddings.
)

minibatch_size = 64


class FilmRecommenderNet(nn.Module):
    def __init__(
        self,
        num_genres,
        num_movies,
        user_context_size,
        item_feature_embedding_size=25,
        item_movieId_embedding_size=25,
        user_feature_embedding_size=50,
    ):
        super().__init__()

        # Movie genre feature tower
        self.i_W1 = nn.Linear(num_genres, item_feature_embedding_size)

        # Movie ID embedding tower (using nn.Embedding)
        self.movie_embedding = nn.Embedding(num_movies, item_movieId_embedding_size)
        self.e_W1 = nn.Linear(item_movieId_embedding_size, item_movieId_embedding_size)

        # User feature tower
        self.u_W1 = nn.Linear(user_context_size, user_feature_embedding_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize with small random values
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.1)
                nn.init.normal_(module.bias, mean=0.0, std=0.1)
            elif isinstance(module, nn.Embedding):
                nn.init.uniform_(module.weight, a=0.0, b=0.1)

    def forward(self, user_contexts, movie_contexts, movie_ids):
        # User tower
        user_embedding = torch.tanh(self.u_W1(user_contexts))

        # Movie feature tower
        item_feature_embedding = torch.tanh(self.i_W1(movie_contexts))

        # Movie ID embedding tower
        item_id_embedding = self.movie_embedding(movie_ids)
        item_embedding_hidden = torch.tanh(self.e_W1(item_id_embedding))

        # Concatenate item embeddings
        item_embedding_combined = torch.cat(
            (item_feature_embedding, item_embedding_hidden), dim=1
        )

        # Final prediction (dot product)
        preds = torch.einsum("ij,ij->i", user_embedding, item_embedding_combined)

        return preds

    def get_movie_embeddings(self, movie_contexts, movie_ids):
        """Extract movie embeddings for similarity computation"""
        with torch.no_grad():
            # Movie feature embedding
            feature_emb = torch.tanh(self.i_W1(movie_contexts))

            # Movie ID embedding
            id_emb = self.movie_embedding(movie_ids)
            id_emb_hidden = torch.tanh(self.e_W1(id_emb))

            # Combined embedding
            combined_emb = torch.cat((feature_emb, id_emb_hidden), dim=1)

            return {
                "MOVIE_FEATURE_EMBEDDING": feature_emb,
                "MOVIEID_EMBEDDING": id_emb_hidden,
                "MOVIE_EMBEDDING_COMBINED": combined_emb,
            }


# Create the model
model = FilmRecommenderNet(
    num_genres=len(genres),
    num_movies=len(top_movies),
    user_context_size=user_context_size,
    item_feature_embedding_size=item_feature_embedding_size,
    item_movieId_embedding_size=item_movieId_embedding_size,
    user_feature_embedding_size=user_feature_embedding_size,
).to(DEVICE)

# Use PyTorch optimizer instead of manual updates
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.5)

# Loss function
loss_fn = torch.nn.MSELoss()

print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters())}")


def train_model(
    model,
    X_train,
    Y_train,
    target_movieId_train,
    target_movieId_context_train,
    X_val,
    Y_val,
    target_movieId_val,
    target_movieId_context_val,
    num_epochs=50_000,
    minibatch_size=64,
    log_every=1000,
):
    """Train the film recommender model"""

    # Use PyTorch optimizer instead of manual updates
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.5)

    # Loss function
    loss_fn = torch.nn.MSELoss()

    # Training loop
    loss_train = []
    loss_val = []

    print("Starting training...")
    for i in trange(num_epochs, desc="Training", unit="step"):
        is_full_val_run = i % log_every == 0

        # Select data
        if is_full_val_run:
            X, Y = X_val, Y_val
            target_movieId, target_movieId_context = (
                target_movieId_val,
                target_movieId_context_val,
            )
        else:
            X, Y = X_train, Y_train
            target_movieId, target_movieId_context = (
                target_movieId_train,
                target_movieId_context_train,
            )

        # Create minibatch
        if is_full_val_run:
            ix = torch.arange(X.shape[0], device=DEVICE)
        else:
            ix = torch.randint(0, X.shape[0], (minibatch_size,), device=DEVICE)

        # Forward pass
        if is_full_val_run:
            model.eval()
            with torch.no_grad():
                preds = model(X[ix], target_movieId_context[ix], target_movieId[ix])
                output = loss_fn(preds, Y[ix])
        else:
            model.train()
            optimizer.zero_grad()
            preds = model(X[ix], target_movieId_context[ix], target_movieId[ix])
            output = loss_fn(preds, Y[ix])
            output.backward()
            optimizer.step()
            scheduler.step()

        # Logging
        if is_full_val_run:
            loss_val.append(output.item())
            if i >= log_every:
                avg_train_loss = np.mean(loss_train[i - log_every : i])
            else:
                avg_train_loss = output.item()
            print(f"[TRAIN] i: {i} | loss: {avg_train_loss:.4f}")
            print(f"[VAL] i: {i} | loss: {output.item():.4f}")
            print()
        else:
            loss_train.append(output.item())

    return loss_train, loss_val


# Check if model already exists
model_path = "film_recommender_model.pth"
if os.path.exists(model_path):
    print(f"Loading existing model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print("Model loaded successfully!")
else:
    print("No existing model found. Training new model...")
    # Preprocess dataset
    user_to_movie_to_rating_LABEL, user_to_context, user_to_avg_rating = (
        preprocess_dataset(df_ratings_aggregated)
    )
    # Split dataset into train and validation sets

    (
        X_train,
        Y_train,
        target_movieId_train,
        target_movieId_context_train,
        X_val,
        Y_val,
        target_movieId_val,
        target_movieId_context_val,
    ) = train_test_split(
        user_to_movie_to_rating_LABEL,
        user_to_context,
        item_emb_movieId_to_i,
        movieId_to_context,
        user_to_avg_rating,
    )
    loss_train, loss_val = train_model(
        model,
        X_train,
        Y_train,
        target_movieId_train,
        target_movieId_context_train,
        X_val,
        Y_val,
        target_movieId_val,
        target_movieId_context_val,
    )

    # Save the trained model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

# Extract embeddings using the model
movieId_to_embedding = {}

print("Extracting movie embeddings...")
model.eval()

for movieId in tqdm(top_movies, desc="Extracting embeddings", unit="movie"):
    movie_idx = torch.tensor([item_emb_movieId_to_i[movieId]], device=DEVICE)
    movie_context = torch.tensor([movieId_to_context[movieId]], device=DEVICE)

    embeddings = model.get_movie_embeddings(movie_context, movie_idx)
    movieId_to_embedding[movieId] = embeddings


# Vectorized similarity computation (much faster!)
def compute_similarities_vectorized(movieId_to_embedding, top_movies, k=5):
    """Compute similarities using vectorized operations"""
    similarities = {}

    embedding_types = [
        "MOVIE_FEATURE_EMBEDDING",
        "MOVIEID_EMBEDDING",
        "MOVIE_EMBEDDING_COMBINED",
    ]

    for emb_type in tqdm(
        embedding_types, desc="Computing similarities", unit="embedding_type"
    ):
        # Stack all embeddings for this type
        embeddings = torch.stack(
            [
                movieId_to_embedding[movieId][emb_type].squeeze(0)
                for movieId in top_movies
            ]
        )

        # Compute pairwise distances using torch.cdist
        distances = torch.cdist(embeddings, embeddings)

        # Get top-k similar movies for each movie
        topk_distances, topk_indices = torch.topk(
            distances, k=k + 1, largest=False
        )  # +1 to exclude self

        for i, movieId in enumerate(top_movies[:15]):  # Only for first 5 movies
            if movieId not in similarities:
                similarities[movieId] = {}

            # Skip the first result (distance to self = 0)
            similar_indices = topk_indices[i][1 : k + 1].cpu().numpy()
            similar_distances = topk_distances[i][1 : k + 1].cpu().numpy()

            similarities[movieId][emb_type] = [
                (top_movies[idx], dist)
                for idx, dist in zip(similar_indices, similar_distances)
            ]

    return similarities


# Compute similarities (much faster than nested loops!)
movieId_to_emb_type_to_similarities = compute_similarities_vectorized(
    movieId_to_embedding, top_movies
)

# Print results
print("Top 5 similar movies for each embedding type:")
for movieId in movieId_to_emb_type_to_similarities.keys():
    print(f"Movie: {movieId_to_title[movieId]}")
    for emb_type, similarities in movieId_to_emb_type_to_similarities[movieId].items():
        print(f"  Embedding Type: {emb_type}")
        for similar_movie, distance in similarities:
            print(
                f"    Similar Movie: {movieId_to_title[similar_movie]} | Distance: {distance:.4f}"
            )
    print()
