from datetime import datetime
import os

import numpy as np
import torch
import torch.optim as optim
from allennlp.common.util import dump_metrics
from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import \
    StanfordSentimentTreeBankDatasetReader
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers.elmo_indexer import \
    ELMoTokenCharactersIndexer
from allennlp.data.vocabulary import Vocabulary
#from codeml.predictors import SentenceClassifierPredictor
from allennlp.models.archival import archive_model
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.seq2vec_encoders.cnn_encoder import CnnEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.training.trainer import Trainer

from codeml.models import BasicClassifier, SequenceClassifier

EMBEDDING_DIM = 128
HIDDEN_DIM = 128


def main():
    # In order to use ELMo, each word in a sentence needs to be indexed with
    # an array of character IDs.
    elmo_token_indexer = ELMoTokenCharactersIndexer()
    reader = StanfordSentimentTreeBankDatasetReader(
        token_indexers={'elmo': elmo_token_indexer})

    train_dataset = reader.read(
        'data/stanfordSentimentTreebank/trees/train.txt')
    dev_dataset = reader.read('data/stanfordSentimentTreebank/trees/dev.txt')

    # Use the 'Small' pre-trained model
    # options_file = ('.pretrained/elmo/small/elmo_2x1024_128_2048cnn_1xhighway_options.json')
    # weight_file = ('.pretrained/elmo/small/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5')
    # or

    options_file = ('https://s3-us-west-2.amazonaws.com/allennlp/models/elmo'
                    '/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json')
    weight_file = ('https://s3-us-west-2.amazonaws.com/allennlp/models/elmo'
                   '/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5')

    elmo_embedder = ElmoTokenEmbedder(options_file, weight_file)

    vocab = Vocabulary.from_instances(train_dataset + dev_dataset)

    word_embeddings = BasicTextFieldEmbedder({"elmo": elmo_embedder})

    elmo_embedding_dim = 256
    cnn = CnnEncoder(elmo_embedding_dim, num_filters=100,
                     ngram_filter_sizes=(2, 3, 4, 5), output_dim=256)

    model = BasicClassifier(vocab, word_embeddings, cnn)
    optimizer = optim.Adam(model.parameters())

    iterator = BucketIterator(batch_size=32, sorting_keys=[
                              ("tokens", "num_tokens")])

    iterator.index_with(vocab)

    serialization_dir = '.experiments/'+datetime.now().strftime("%m_%d_%H_%M_%S")
    if not os.path.exists(serialization_dir):
        os.makedirs(serialization_dir)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=dev_dataset,
                      patience=5,
                      num_epochs=10,
                      serialization_dir=serialization_dir)

    metrics = trainer.train()
    dump_metrics(os.path.join(serialization_dir,
                              "metrics.json"), metrics, log=True)


if __name__ == '__main__':
    main()
