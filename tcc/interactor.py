class ClassificationException(Exception):
    pass

def classify(comment, model=None, X_train=None, y_train=None):
    if model and X_train and y_train:
        model.train(X_train, y_train)
        return model.predict(comment)
    elif model:
        model.load_default()
        return model.predict(comment)
    else:
        raise ClassificationException("No model or training data specified")