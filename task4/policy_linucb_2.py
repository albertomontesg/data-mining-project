import numpy as np

SIGMA = 0.2     # SIGMA in (0, 1) with confidence 1 - SIGMA
ALPHA = 1 + np.sqrt(np.log(2 / SIGMA) / 2)

DIMENSION = 6


M = dict()  # Key: article ID. Value: matrix M for LinUCB algorithm.
M_inv = dict() # Key: article ID. Value: inverted matrix M.
b = dict()  # Key: article ID. Value: number b for LinUCB algorithm.
w = dict()  # Key: article ID. Value: weights w for LinUCB algorithm.
# ucb_value = dict()  # Key: article id. Value: ucb value for LinUCB algorithm.

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
        M[article_id] = np.identity(DIMENSION)
        M_inv[article_id] = np.identity(DIMENSION)
        b[article_id] = np.zeros((DIMENSION, 1))
        w[article_id] = np.zeros((DIMENSION, 1))


def update(reward):
    """Update our model given that we observed 'reward' for our last recommendation."""

    if reward == -1:    # If the log file did not have matching recommendation
        return
    else:
        # Update M, b and weights
        M[last_article_id] += last_user_features.dot(last_user_features.T)
        M_inv[last_article_id] = np.linalg.inv(M[last_article_id])
        b[last_article_id] += reward * last_user_features
        w[last_article_id] = M_inv[last_article_id].dot(b[last_article_id])    # Update weight


def recommend(time, user_features, articles):
    """Recommend an article."""
    best_article_id = None
    best_ucb_value = -1

    user_features = np.asarray(user_features)
    user_features.shape = (DIMENSION, 1)

    for article_id in articles:
        # If we haven't seen article before
        if article_id not in M:
            # Initialise this article's variables
            M[article_id] = np.identity(DIMENSION)
            M_inv[article_id] = np.identity(DIMENSION)
            b[article_id] = np.zeros((DIMENSION, 1))
            w[article_id] = np.zeros((DIMENSION, 1))

            # Get at least 1 datapoint for this article
            best_article_id = article_id
            break

        # If we have seen article before
        else:
            ucb_value = w[article_id].T.dot(user_features) +\
                        ALPHA * np.sqrt(user_features.T.dot(M_inv[article_id]).dot(user_features))

            if ucb_value > best_ucb_value:
                best_ucb_value = ucb_value
                best_article_id = article_id

    global last_article_id
    last_article_id = best_article_id   # Remember which article we are going to recommend
    global last_user_features
    last_user_features = user_features  # Remember what the user features were

    return best_article_id
