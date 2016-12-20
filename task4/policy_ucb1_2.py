import numpy as np

DIMENSION = 6


mu = dict()  # Key: article ID. Value: mean payoff.
n = dict()  # Key: article ID. Value: number of times observed.

article_list = None

# Remember last article and user so we can use this information in update() function.
last_article_id = None
last_user_features = None

def set_articles(articles):
    """Initialise whatever is necessary, given the articles."""
    global article_list

    # Make a list of article ID-s
    if isinstance(articles, dict):
        article_list = [x for x in articles]        # If 'articles' is a dict, get all the keys
    else:
        article_list = [x[0] for x in articles]     # If 'articles' is a matrix, get 1st element from each row

    for article_id in article_list:
        # Initialise M and b
        mu[article_id] = 0
        n[article_id] = 0


def update(reward):
    """Update our model given that we observed 'reward' for our last recommendation."""

    if reward == -1:    # If the log file did not have matching recommendation
        return
    else:
        # Update
        n[last_article_id] += 1
        mu[last_article_id] += 1.0 / n[last_article_id] * (reward - mu[last_article_id])

def recommend(time, user_features, articles):
    """Recommend an article."""
    best_article_id = None
    best_ucb_value = -1

    user_features = np.asarray(user_features)
    user_features.shape = (DIMENSION, 1)

    for article_id in articles:
        # If we haven't seen article before
        if article_id not in mu or n[article_id] == 0:
            # Initialise this article's variables
            mu[article_id] = 0
            n[article_id] = 0

            # Get at least 1 datapoint for this article
            best_article_id = article_id
            break

        # If we have seen article before
        else:
            ucb_value = mu[article_id] + np.sqrt(2 * np.log(time) / n[article_id])

            if ucb_value > best_ucb_value:
                best_ucb_value = ucb_value
                best_article_id = article_id

    global last_article_id
    last_article_id = best_article_id   # Remember which article we are going to recommend
    global last_user_features
    last_user_features = user_features  # Remember what the user features were

    return best_article_id
