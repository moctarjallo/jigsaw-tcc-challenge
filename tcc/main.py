import numpy as np
from helpers import remove_callback_dir, make_callback_dir, save_array

from core import ToxicComments, ToxicModel, prepare




def run():
    # Load data and build model
    batch_size = 10000
    n_features = 100
    tc = ToxicComments('tcc/data/train.csv', batch_size=batch_size)
    model = ToxicModel(n_features, 6)

    # Train model
    remove_callback_dir('events')
    remove_callback_dir('ckpts')
    iteration = 0
    for _, comments, labels in tc:
        comments = np.array(prepare(comments, n_features))
        labels = np.array(labels)

        iteration+=1
        events_dir = make_callback_dir('events', iteration)
        ckpts_dir = make_callback_dir('ckpts', iteration)

        model.train(comments, labels, callback_dirs=[events_dir, ckpts_dir])

    # Test model
    tc_test = ToxicComments('tcc/data/test.csv')
    ids, comments, _ = next(tc_test)
    comments = prepare(comments, n_features=n_features)
    predictions = model.predict(comments)
    save_array(ids, predictions, 'tcc/data/test_submission5.csv')



if __name__ == '__main__':
    run()
