import json
import sys
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet
from tqdm import tqdm
import spacy

nlp = spacy.load("en_core_web_sm")

def add_negation(text, inconsistent_list):
    # Parse the sentence using SpaCy
    doc = nlp(text)
    negated = False

    i = 0
    while i < len(doc):
        token = doc[i]
        if not negated and token.text in ['have', 'has'] and doc[i-1].text not in ["not", "n't"]:
            # Check if the verb has a direct object (dobj) which is a noun
            for child in token.children:
                if child.dep_ == 'dobj' and child.pos_ in ['NOUN', 'PRON', 'PROPN']:
                    # Create the negated sentence
                    if token.text == 'have':
                        negated_sentence = text[:token.idx] + 'do not have' + text[token.idx+len(token.text):]
                        inconsistent_list.append(negated_sentence)
                        negated = True
                    elif token.text == 'has':
                        negated_sentence = text[:token.idx] + 'does not have' + text[token.idx+len(token.text):]
                        inconsistent_list.append(negated_sentence)
                        negated = True
                    break
        
        elif not negated and token.text in ['is', 'are'] and i + 1 < len(doc) and doc[i+1].text not in ['not',
                                                                                                        "n't",
                                                                                                        'no',
                                                                                                        'nothing']:
            if token.text == 'is':
                negated_sentence = text[:token.idx] + 'is not' + text[token.idx+len(token.text):]
                inconsistent_list.append(negated_sentence)
                negated = True
            elif token.text == 'are':
                negated_sentence = text[:token.idx] + 'are not' + text[token.idx+len(token.text):]
                inconsistent_list.append(negated_sentence)
                negated = True

            elif token.i == len(doc) - 1:
                if token.text == 'is':
                    negated_sentence = text[:token.idx] + 'is not' + text[token.idx+len(token.text):]
                    inconsistent_list.append(negated_sentence)
                    negated = True
                elif token.text == 'are':
                    negated_sentence = text[:token.idx] + 'are not' + text[token.idx+len(token.text):]
                    inconsistent_list.append(negated_sentence)
                    negated = True

        i += 1
    return inconsistent_list

def remove_negation(text, inconsistent_list):
    # Parse the sentence using SpaCy
    doc = nlp(text)
    negation_removed = False

    i = 0
    while i < len(doc):
        token = doc[i]
        # Check for "does not have" or "do not have" followed by a noun
        if not negation_removed and token.text in ["does", "do"] and i + 2 < len(doc) and doc[i+1].text in ["not", "n't"] and doc[i+2].text == "have":
            for child in doc[i+2].children:
                if child.dep_ == 'dobj' and child.pos_ in ['NOUN', 'PRON', 'PROPN']:
                    if token.text == 'do':
                        negated_sentence = text[:token.idx] + 'have' + text[doc[i+2].idx+len(doc[i+2]):]
                        inconsistent_list.append(negated_sentence)
                        negation_removed = True
                        i += 3
                    elif token.text == 'does':
                        negated_sentence = text[:token.idx] + 'has' + text[doc[i+2].idx+len(doc[i+2]):]
                        inconsistent_list.append(negated_sentence)
                        negation_removed = True
                        i += 3
                    break

        elif not negation_removed and token.text in ["is", "are"] and i + 2 < len(doc) and doc[i+1].text in ["not",
                                                                                                             "no",
                                                                                                             "n't"]:
            if token.text == 'is':
                negated_sentence = text[:token.idx] + 'is' + text[doc[i+1].idx+len(doc[i+1]):]
                inconsistent_list.append(negated_sentence)
                negation_removed = True
                i += 2
            elif token.text == 'are':
                negated_sentence = text[:token.idx] + 'are' + text[doc[i+1].idx+len(doc[i+1]):]
                inconsistent_list.append(negated_sentence)
                negation_removed = True
                i += 2
        
        i += 1
    
    return inconsistent_list

class word_antonym_replacer(object):
    def replace(self, word, pos=None):
        antonyms = set()
        for syn in wordnet.synsets(word, pos=pos):
            for lemma in syn.lemmas():
                for antonym in lemma.antonyms():
                    antonyms.add(antonym.name())

        if len(antonyms) > 0:
            return antonyms
        else:
            return None

def replace_adjectives_adverbs(text, inconsistent_list):
    # Tokenize the text sentence
    tokens = word_tokenize(text)
    # Get the part-of-speech tags for the tokens
    pos_tags = pos_tag(tokens)
    word_list = []
    pos_list = []
    for word, pos in pos_tags:
        word_list.append(word)
        pos_list.append(pos)
    
    rep_antonym = word_antonym_replacer()
    if 'JJ' in pos_list:
        replaces = rep_antonym.replace(word_list[pos_list.index('JJ')], wordnet.ADJ)
        if replaces is not None:
            for replace in replaces:
                inconsistent_list.append(text.replace(word_list[pos_list.index('JJ')], replace, 1))
    if 'RB' in pos_list:
        replaces = rep_antonym.replace(word_list[pos_list.index('RB')], wordnet.ADV)
        if replaces is not None:
            for replace in replaces:
                inconsistent_list.append(text.replace(word_list[pos_list.index('RB')], replace, 1))
    return inconsistent_list

if __name__ == '__main__':
    data = json.load(open(sys.argv[1], 'r'))
    ids_list = list(data.keys())
    
    for idx in tqdm(ids_list):
        inconsistent_list = []
        for text in data[idx]['explanation']:
            #inconsistent_list = add_negation(text, inconsistent_list)
            #inconsistent_list = remove_negation(text, inconsistent_list)
            inconsistent_list = replace_adjectives_adverbs(text, inconsistent_list)
        data[idx]['explanation'] = inconsistent_list

        if not data[idx]['explanation']:
            data.pop(idx, None)


    with open(sys.argv[2], 'w') as file:
        json.dump(data, file, indent=4)