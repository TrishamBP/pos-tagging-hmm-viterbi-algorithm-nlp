from utils import get_word_tag, preprocess, create_dictionaries, compute_accuracy
from hidden_markov_model import create_transition_matrix, create_emission_matrix
from viterbi import initialize, viterbi_forward, viterbi_backward

# Data Sources
# load in the training corpus
with open("./data/WSJ_02-21.pos", 'r') as f:
    training_corpus = f.readlines()
# read the vocabulary data, split by each line of text, and save the list
with open("./data/hmm_vocab.txt", 'r') as f:
    voc_l = f.read().split('\n')
# vocab: dictionary that has the index of the corresponding words
vocab = {}

# Get the index of the corresponding words.
for i, word in enumerate(sorted(voc_l)):
    vocab[word] = i
# load in the test corpus
with open("./data/WSJ_24.pos", 'r') as f:
    y = f.readlines()
# corpus without tags, preprocessed
_, prep = preprocess(vocab, "./data/test.words")
emission_counts, transition_counts, tag_counts = create_dictionaries(training_corpus, vocab)
# get all the POS states
states = sorted(tag_counts.keys())

# Hidden Markov Models for POS
alpha = 0.001
A = create_transition_matrix(alpha, tag_counts, transition_counts)
# creating your emission probability matrix. this takes a few minutes to run.
alpha = 0.001
B = create_emission_matrix(alpha, tag_counts, emission_counts, list(vocab))

# Viterbi Algorithm and Dynamic Programming
best_probs, best_paths = initialize(states, tag_counts, A, B, prep, vocab)
best_probs, best_paths = viterbi_forward(A, B, prep, best_probs, best_paths, vocab)
pred = viterbi_backward(best_probs, best_paths, prep, states)
m = len(pred)

print(f"Accuracy of the Viterbi algorithm is {compute_accuracy(pred, y):.4f}")

