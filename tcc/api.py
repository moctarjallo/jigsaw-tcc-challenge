from tcc.interactor import classify, ClassificationException


def classifier(comments, processor, model=None, data=None) :
    if data:
        training_comments = processor(data['comments']).words
        training_labels = data['labels']
    comments = processor(comments['comments']).words
    try:
        predictions = classify(comments.norm, model=model, X_train=training_comments.norm, y_train=training_labels)
        return predictions
    except ClassificationException as e:
        raise e

def saver(data):
    pass


