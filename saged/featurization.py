"""Functions to create various types of features, combine them, and create metafeatures."""

import numbers
import string
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from saged.external_libraries.features.Word2VecFeatures import Word2VecFeatures

def create_tf_idf(column, chars=None):
    """Creates TF-IDF features of the given column.

    The chars parameter specifies which characters should appear in the resulting dataframe. If
    the data covers characters that are not present in chars, the associated features will be
    dropped. If there are more characters in chars than in the data, zeros will be added in place
    of those features.

    If chars is not specified, list(string.printable) is used (printable ASCII characters).

    Parameters:
        column (pandas.core.series.Series): The column to create features of.
        chars (list[str], optional): A list of characters. Defaults to list(string.printable).

    Returns:
        pandas.core.frame.DataFrame: The resulting feature DataFrame (the columns' order equals the
            order of chars).
    """
    if chars is None:
        chars = list(string.printable)

    pipeline = Pipeline([
        ('vect', CountVectorizer(analyzer='char', lowercase=False, ngram_range=(1, 1))),
        ('tfidf', TfidfTransformer())
    ])

    data = column.astype(str)
    tf_idf = pipeline.fit_transform(data)
    features = pd.DataFrame(tf_idf.toarray(), columns=pipeline.get_feature_names_out())

    padded_features = pd.DataFrame(np.zeros((len(column), len(chars))), columns=chars)
    padded_features.update(features)
    return padded_features

def create_w2v(df, vector_size=100, epochs=10):
    """Creates Word2Vec features of the given dataframe.

    Currently, this is simply a port of the legacy module
    saged.external_libraries.features.Word2VecFeatures.py. (TODO rewrite)

    Parameters:
        df (pandas.core.frame.DataFrame): The dataframe to create features of.
        vector_size (int, optional): Vector size for Word2Vec vectors. Defaults to 100.
        epochs (int, optional): Number of epochs to train the Word2Vec model for. Defaults to 10.

    Returns:
        pandas.core.frame.DataFrame: The resulting feature DataFrame.
    """
    model = Word2VecFeatures(vector_size=vector_size, epochs=epochs)
    features, feature_names, _ = model.add_word2vec_features(df, None, [])

    return pd.DataFrame(features.toarray(), columns=feature_names)

def create_metadata(column):
    """Creates metadata of the given column, that is:
        * number of occurences of value
        * 1 if value is numeric, else 0
        * (string) length of value
        * 1 if value is alphabetic, else 0
        * value if it is numeric, else 0

    Parameters:
        column (pandas.core.series.Series): The column to create features of.

    Returns:
        pandas.core.frame.DataFrame: The resulting feature DataFrame.
    """
    # TODO maybe NaN values should be dropped?
    value_counts = column.value_counts(dropna=False)

    return pd.DataFrame(np.array([
            column.map(lambda x: value_counts[x]),
            column.map(lambda x: isinstance(x, numbers.Number)),
            column.astype(str).map(len),
            column.astype(str).map(str.isalpha),
            column.map(lambda x: x if isinstance(x, numbers.Number) and not np.isnan(x) else 0)
        ]).T,
        columns=["num_occurences", "is_numeric", "length", "is_alphabetical", "extracted_number"],
    )

def create_features(df, tf_idf_chars=None, w2v_size=100):
    """Creates TF-IDF, Word2Vec and metadata features of the given dataset.

    Parameters:
        df (pandas.core.frame.DataFrame): The dataframe to create features of
        tf_idf_chars (list[str], optional): Characters to use for the TF-IDF features, defaults
            to list(string.printable) (see create_tf_idf()).
        w2v_size (int, optional): Vector size for Word2Vec vectors. Defaults to 100.

    Returns:
        dict[pandas.core.frame.DataFrame]: A dictionary mapping the given dataframe's column names
            to pandas DataFrames containing the created features.
    """
    features = {}

    # Word2Vec features require the full dataframe, not just one column
    w2v_all = create_w2v(df, vector_size=w2v_size)

    # The other features are created separately
    for column in df:
        tf_idf = create_tf_idf(df[column], chars=tf_idf_chars)
        metadata = create_metadata(df[column])

        # Extract Word2Vec features: the column names have the format <column-name>_word2vec_<i>
        w2v = w2v_all[[col for col in w2v_all if col.startswith(column + "_word2vec")]]
        w2v.columns = w2v.columns.str.removeprefix(column + "_")

        features[column] = pd.concat([tf_idf, w2v, metadata], axis="columns")

    return features
