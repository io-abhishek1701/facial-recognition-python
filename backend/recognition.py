import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


SIMILARITY_THRESHOLD = 0.70


def compare_embeddings(new_embedding, stored_embedding):
    """
    Compare two embeddings using cosine similarity.
    Returns the similarity score.
    """

    score = cosine_similarity(
        [new_embedding],
        [stored_embedding]
    )[0][0]

    return float(score)


def is_match(score):
    """
    Returns True if the similarity score is above the threshold.
    """

    return score >= SIMILARITY_THRESHOLD