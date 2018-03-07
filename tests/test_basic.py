# -*- coding: utf-8 -*-

from context import ToxicComments, prepare, ToxicModel, save_array

import unittest


# class TestPreprocessor(unittest.TestCase):
#     def test_toarray(self):
#         # doc = ["Hello this is just a simple test document"]
#         # preprocessor = Preprocessor(doc)
#         # print(preprocessor.toarray())
#         pass

class TestToxicComments(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.tc = ToxicComments('tcc/data/sample.csv', batch_size=self.batch_size)
        self.ids = ["0000997932d777bf", "000103f0d9cfb60f", "000113f07ec002fd", "0001b41b1c6bb37e", "0001d958c54c6e35", "00025465d4725e87", "0002bcb3da6cb337", "00031b1e95af7921", "00037261f536c51d", "00040093b2687caa"]
        self.comments = ["""Explanation\nWhy the edits made under my username Hardcore Metallica Fan were reverted? They weren't vandalisms, just closure on some GAs after I voted at New York Dolls FAC. And please don't remove the template from the talk page since I'm retired now.89.205.38.27""",\
        """D'aww! He matches this background colour I'm seemingly stuck with. Thanks.  (talk) 21:51, January 11, 2016 (UTC)""",\
        "Hey man, I'm really not trying to edit war. It's just that this guy is constantly removing relevant information and talking to me through edits instead of my talk page. He seems to care more about the formatting than the actual info.",\
        """"
More
I can't make any real suggestions on improvement - I wondered if the section statistics should be later on, or a subsection of \"\"types of accidents\"\"  -I think the references may need tidying so that they are all in the exact same format ie date format etc. I can do that later on, if no-one else does first - if you have any preferences for formatting style on references or want to do it yourself please let me know.

There appears to be a backlog on articles for review so I guess there may be a delay until a reviewer turns up. It's listed in the relevant form eg Wikipedia:Good_article_nominations#Transport  \"""",\
        "You, sir, are my hero. Any chance you remember what page that's on?",\
        """"

Congratulations from me as well, use the tools well.  · talk \"""",\
        "COCKSUCKER BEFORE YOU PISS AROUND ON MY WORK",\
        "Your vandalism to the Matt Shirvington article has been reverted.  Please don't do it again, or you will be banned.",\
        "Sorry if the word 'nonsense' was offensive to you. Anyway, I'm not intending to write anything in the article(wow they would jump on me for vandalism), I'm merely requesting that it be more encyclopedic so one can use it for school as a reference. I have been to the selective breeding page but it's almost a stub. It points to 'animal breeding' which is a short messy article that gives you no info. There must be someone around with expertise in eugenics? 93.161.107.169",\
        "alignment on this subject and which are contrary to those of DuLithgow"]

        self.labels = [[0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0],\
         [0,0,0,0,0,0], [1,1,1,0,1,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0]]


    # def test_ids(self):
    #     self.assertEqual(self.tc.ids, self.ids[:self.batch_size])
    #
    #
    # def test_comments(self):
    #     self.assertEqual(self.tc.comments, self.comments[:self.batch_size])
    #
    # def test_labels(self):
    #     self.assertEqual(self.tc.labels, self.labels[:self.batch_size])

    def test_next(self):
        self.assertEqual(next(self.tc), (self.ids[:self.batch_size],
                                         self.comments[:self.batch_size],
                                         self.labels[:self.batch_size]))
        self.assertEqual(next(self.tc), (self.ids[self.batch_size:2*self.batch_size],
                                         self.comments[self.batch_size:2*self.batch_size],
                                         self.labels[self.batch_size:2*self.batch_size]))
        self.assertEqual(next(self.tc), (self.ids[2*self.batch_size:3*self.batch_size],
                                         self.comments[2*self.batch_size:3*self.batch_size],
                                         self.labels[2*self.batch_size:3*self.batch_size]))

    # def test_prepare(self):
    #     print(prepare(next(self.tc)[1]))


class TestToxicModel(unittest.TestCase):
    def setUp(self):
        self.batch_size = 10000
        self.n_features = 30
        self.model = ToxicModel(self.n_features, 6)

    def test_load_weights(self):
        self.model.load('tcc/weigths09-0.2938.hdf5')
        tc = ToxicComments('tcc/data/test.csv')
        ids, comments, _ = next(tc)
        comments  = prepare(comments, self.n_features)
        predictions = self.model.predict(comments)
        # print(predictions)
        save_array(ids, predictions, 'tcc/data/test_submission1.csv')





if __name__ == '__main__':
    unittest.main()
