import datasets
from transformers import AutoTokenizer, TFDistilBertForTokenClassification
import tensorflow as tf

class Trainer:

    def __init__(self) -> None:
        pass

    def train(self):
        pass

    def load_model(self):

        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = TFDistilBertForTokenClassification.from_pretrained("distilbert-base-uncased")
        inputs = tokenizer(
            "HuggingFace is a company based in Paris and New York", add_special_tokens=False, return_tensors="tf"
        )
        logits = model(**inputs).logits
        predicted_token_class_ids = tf.math.argmax(logits, axis=-1)
        # Note that tokens are classified rather then input words which means that
        # there might be more predicted token classes than words.
        # Multiple token classes might account for the same word
        predicted_tokens_classes = [model.config.id2label[t] for t in predicted_token_class_ids[0].numpy().tolist()]
        print(f'predicted_tokens_classes: {predicted_tokens_classes}')


    def data_loader(self):
        '''
        '''
        finer_train = datasets.load_dataset("nlpaueb/finer-139", split="train")
        print(f'finer_train: {finer_train}')
        #print(f'finer_train.features["id"]: {finer_train.features["id"].feature.names}')
        #print(f'finer_train.features["tokens"]: {finer_train.features["tokens"].feature.names}')
        print(f'finer_train.features["ner_tags"]: {finer_train.features["ner_tags"].feature.names}')

        #finer_tag_names = finer_train.features["ner_tags"].feature.names
        #print(f'finer_tag_names: {finer_tag_names}')



if __name__ == "__main__":
    print(f'Welcome to trainer')
    TRAIN = Trainer()
    TRAIN.data_loader()
    #TRAIN.load_model()
