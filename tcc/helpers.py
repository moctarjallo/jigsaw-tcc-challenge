import os
import pandas as pd

"""Remove a directory tree"""
def remove_callback_dir(directory):
    from shutil import rmtree
    to_remove = os.path.join('tcc', directory)
    if os.path.exists(to_remove):
        rmtree(to_remove)

"""Create a directory tree"""
def make_callback_dir(directory, i):
    full_dir = os.path.join('tcc', directory, str(i))
    if not os.path.exists(full_dir):
        os.makedirs(full_dir)
    return full_dir

"""Save array data in csvfile with ids=ids"""
def save_array(ids, array, csvfile):
    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    pd.DataFrame(array, index=ids, columns=labels).to_csv(csvfile)
