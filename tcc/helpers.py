import os
import pandas as pd

def remove_callback_dir(directory):
    from shutil import rmtree
    to_remove = os.path.join('tcc', directory)
    if os.path.exists(to_remove):
        rmtree(to_remove)

def make_callback_dir(directory, i):
    full_dir = os.path.join('tcc', directory, str(i))
    if not os.path.exists(full_dir):
        os.makedirs(full_dir)
    return full_dir

def save(ids, predictions, csvfile):
    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    pd.DataFrame(predictions, index=ids, columns=labels).to_csv(csvfile)
