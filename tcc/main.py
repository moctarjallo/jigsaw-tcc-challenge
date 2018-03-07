import numpy as np
from helpers import remove_callback_dir, make_callback_dir, save

from core import ToxicComments, ToxicModel, prepare




def run():
    # Load data and build model
    batch_size = 20
    n_features = 30
    tc = ToxicComments('tcc/data/sample_train.csv', batch_size=batch_size)
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
    tc_test = ToxicComments('tcc/data/sample_test.csv')
    ids, comments, _ = next(tc_test)
    comments = prepare(comments, n_features=n_features)
    predictions = model.predict(comments)
    save(ids, predictions, 'tcc/data/sample_submission_test.csv')

run()
