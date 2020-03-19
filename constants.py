# Categorical features are assumed to each have a maximum value in the dataset.
MAX_CATEGORICAL_FEATURE_VALUES = [24, 31, 12]

CATEGORICAL_FEATURE_KEYS = ['Pclass', 'Parch', 'SibSp']

DENSE_FLOAT_FEATURE_KEYS = ['Fare', 'PassengerId', 'Age']

# Number of vocabulary terms used for encoding VOCAB_FEATURES by tf.transform
VOCAB_SIZE = 1000

# Count of out-of-vocab buckets in which unrecognized VOCAB_FEATURES are hashed.
OOV_SIZE = 10

VOCAB_FEATURE_KEYS = ['Embarked', 'Sex', 'Cabin', 'Name', 'Ticket']

# Keys
LABEL_KEY = 'Survived'


def transformed_name(key):
    return key + '_xf'
