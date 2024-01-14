import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load data
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Check if the data needs padding
def pad_sequences(sequences, maxlen=None, dtype='float32', padding='post', truncating='post', value=0.0):
    # If sequences are lists, convert them to numpy arrays
    if isinstance(sequences[0], list):
        sequences = [np.array(s) for s in sequences if len(s) > 0]

    lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)

    if maxlen is None:
        maxlen = np.max(lengths)

    # The shape after the first dimension of the sequence arrays
    sample_shape = np.asarray(sequences[0]).shape[1:]

    # Initialize an empty array to hold the padded data
    num_samples = len(sequences)
    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)

    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # Skip empty lists/arrays
        trunc = np.asarray(s, dtype=dtype)

        if truncating == 'pre':
            trunc = trunc[-maxlen:]
        else:
            trunc = trunc[:maxlen]

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        else:
            x[idx, -len(trunc):] = trunc

    return x

# Pad the data
data_padded = pad_sequences(data_dict['data'])

# Convert labels to numpy array
labels = np.asarray(data_dict['labels'])

# Split the data
x_train, x_test, y_train, y_test = train_test_split(data_padded, labels, test_size=0.2, shuffle=True, stratify=labels)

# Create and train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predict and calculate accuracy
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
