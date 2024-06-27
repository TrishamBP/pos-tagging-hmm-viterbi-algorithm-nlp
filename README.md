# Part-of-Speech Tagging with Hidden Markov Models

This project implements a Part-of-Speech (POS) tagger using Hidden Markov Models (HMM) and the Viterbi algorithm. The system is trained on the Wall Street Journal (WSJ) corpus and evaluated on a separate test set.

## Algorithm Details

### Hidden Markov Model

The POS tagger uses a Hidden Markov Model, which consists of:

1. **States**: The possible POS tags (e.g., Noun, Verb, Adjective, etc.)
2. **Observations**: The words in the corpus
3. **Transition Matrix (A)**: Probabilities of transitioning from one POS tag to another
4. **Emission Matrix (B)**: Probabilities of a word being associated with a particular POS tag

#### Transition Matrix Creation
- Function: `create_transition_matrix(alpha, tag_counts, transition_counts)`
- Smoothing is applied using the parameter `alpha` to handle unseen transitions

#### Emission Matrix Creation
- Function: `create_emission_matrix(alpha, tag_counts, emission_counts, vocab)`
- Smoothing is also applied here to handle unseen word-tag pairs

### Viterbi Algorithm

The Viterbi algorithm is used for decoding - finding the most likely sequence of POS tags for a given sentence.

1. **Initialization**: 
   - Function: `initialize(states, tag_counts, A, B, corpus, vocab)`
   - Sets up the initial probabilities for the first word in the sequence

2. **Forward Algorithm**:
   - Function: `viterbi_forward(A, B, test_corpus, best_probs, best_paths, vocab)`
   - Computes the most likely path to each state for each word in the sequence

3. **Backward Algorithm**:
   - Function: `viterbi_backward(best_probs, best_paths, corpus, states)`
   - Backtracks to find the most likely sequence of POS tags

### Handling Unknown Words

- Unknown words are assigned special tokens based on their morphological features
- Function: `assign_unk(tok)` in `utils.py`

## Dataset Information

### Wall Street Journal (WSJ) Corpus

The project uses the Wall Street Journal corpus, a widely used dataset in Natural Language Processing for POS tagging tasks.

1. **Training Data**: 
   - File: `./data/WSJ_02-21.pos`
   - Contains sentences from sections 02-21 of the WSJ corpus
   - Format: One word per line, with its corresponding POS tag

2. **Test Data**:
   - Files: 
     - `./data/WSJ_24.pos` (words with tags for evaluation)
     - `./data/test.words` (words only, for tagging)
   - Contains sentences from section 24 of the WSJ corpus
   - Used to evaluate the performance of the POS tagger

3. **Vocabulary**:
   - File: `./data/hmm_vocab.txt`
   - Contains the known words used in the model
   - Unknown words in the test set are handled separately

### Data Format

- Each line in the corpus files contains a word and its POS tag, separated by a space
- Sentences are separated by empty lines

### POS Tags

The WSJ corpus uses a rich set of POS tags. Some common tags include:
- NN: Noun, singular or mass
- VB: Verb, base form
- JJ: Adjective
- RB: Adverb
- DT: Determiner
- IN: Preposition or subordinating conjunction

(Note: The full list of tags is extensive and can be found in the Penn Treebank POS tag set documentation)

## Usage

1. Ensure all data files are in the `./data/` directory.
2. Run the main script:

```bash
python main.py
