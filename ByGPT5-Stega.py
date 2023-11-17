from argparse import ArgumentParser
import transformers
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.pipelines.text_generation import TextGenerationPipeline
from transformers.models.t5.configuration_t5 import T5Config
import argparse
from bygpt5 import ByGPT5Config, ByGPT5LMHeadModel, ByGPT5Tokenizer
from transformers import GPT2Tokenizer, GPT2Model
import os
import torch
import numpy as np
import random
from decimal import *
import math
import sys
from nltk.corpus import wordnet
# from nltk import spell
import spacy
import csv
from evaluate import load
# print(torch.__version__)

getcontext().prec = 400
# fix some warnings inside pipeline
# we need to add this to to be able to use ByGPT5 with AutoModel
CONFIG_MAPPING.register(ByGPT5Config.model_type, ByGPT5Config)
TOKENIZER_MAPPING.register(ByGPT5Config, (ByGPT5Tokenizer, None))
MODEL_FOR_CAUSAL_LM_MAPPING.register(ByGPT5Config, ByGPT5LMHeadModel)
MODEL_FOR_CAUSAL_LM_MAPPING.register(T5Config, ByGPT5LMHeadModel)


perplexity = load("perplexity", module_type="metric")

model = ByGPT5LMHeadModel.from_pretrained("nllg/bygpt5-medium-en")
tokenizer = ByGPT5Tokenizer.from_pretrained("nllg/bygpt5-medium-en")

# tokenizer_ppl = GPT2Tokenizer.from_pretrained('gpt2')
# model_ppl = GPT2Model.from_pretrained('gpt2')

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

    def is_prefix(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

def construct_Trie():

    nlp = spacy.load("en_core_web_sm")
    Vocab = set()

    for word in list(nlp.vocab.strings):
        Vocab.add(word.lower())

    # print(len(Vocab))
    # print(Vocab)

    trie = Trie()
    for word in Vocab:
        trie.insert(word)
    return trie
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic= True

def lexicon(DICT, current_node):
    for key in DICT.keys():
        next_char = tokenizer.decode([key]).lower()
        # print(next_char)
        if (next_char.isalpha()) and (next_char not in current_node.children):
            # DICT[key] = 0
            print('OOV character: ', next_char, DICT[key])
            DICT[key] = DICT[key] * Decimal(args.OOV_multiple)

    sum_p = Decimal(0.0)
    for value in DICT.values():
        sum_p += value
    NEW_DICT = dict()
    for key in DICT.keys():
        NEW_DICT[key] = DICT[key] / sum_p

    return NEW_DICT

def generate_next_token(current_min, current_max, new_dict, secret_decimal):
    interval = current_max - current_min
    for t in new_dict.keys():
        current_max = current_min + interval * new_dict[t]
        if current_max >= secret_decimal:
            return t, current_min, current_max
        else:
            current_min = current_max
    print('Error: Not Found!')
    sys.exit()

# print(input_ids)
def generate(input_ids, secret_decimal):
    num = 0
    current_min = Decimal(0.0)
    current_max = Decimal(2 ** args.secret_length)
    Avg_entropy = Decimal(0.0)
    Entropy_list =[]
    trie = construct_Trie()
    current_node = trie.root
    lexicon_flag = True
    OOV_num = 0
    Word_num = 0
    while True:
        # print(input_ids)
        # setup_seed(args.seed)
        output_ids = model.generate(input_ids, max_new_tokens = 1, output_scores = True, return_dict_in_generate=True)
        # setup_seed(args.seed)
        # for c in outputs[0]:
        #     print(c.item())
        # logits = outputs.logits
        # print(logits)
        # print(output_ids['sequences'])
        normalized_probabilities = torch.nn.functional.softmax(output_ids['scores'][0][0] , dim = -1)
        sorted_probabilities, indices = torch.sort(normalized_probabilities, descending=True)
        Entropy = Decimal(0.0)

        for i in range(len(sorted_probabilities)):
            Entropy += Decimal(-sorted_probabilities[i].item()*math.log(sorted_probabilities[i].item(),2))
            # if i < 5:
            #     print('[', format(sorted_probabilities[i].item(), '.4f'),':', format(-math.log(sorted_probabilities[i].item(),2), '.4f') ,']', end=' ')

        # print()
        # print(sorted_probabilities.item())
        # print('Entropy = ',format(Entropy,'.4f'))
        Variance = Decimal(0.0)
        for i in range(len(sorted_probabilities)):
            Variance += Decimal(sorted_probabilities[i].item()*(math.log(sorted_probabilities[i].item(),2)**2))
        Variance -= Entropy**2
        # print(Variance)
        Standard_deviation = float(Variance)**0.5
        # print('Standard deviation = ', format(Standard_deviation,'.4f'))

        # print(indices[0:5])
        # print(sorted_probabilities[0:5])
        filtered_probabilities = []
        # token_probability_dict = dict()
        # sum_p = Decimal(0.0)
        for i in range(len(sorted_probabilities)):
            if -math.log(sorted_probabilities[i].item()) <= Entropy + Decimal(args.sigma_multiple * Standard_deviation):
                # print(i, normalized_probabilities[i])
                # token_probability_dict[i] = Decimal(normalized_probabilities[i].item())
                filtered_probabilities.append(sorted_probabilities[i].item())
            else:
                break
        if lexicon_flag == True:
            Prob_OOV = 0.0
            for i in range(len(filtered_probabilities)):
                temp_character = tokenizer.decode([indices[i]]).lower()
                if (temp_character.isalpha()) and (temp_character not in current_node.children):
                    # print(temp_character)
                    Prob_OOV += filtered_probabilities[i]
            new_temperature = args.max_temperature - (args.max_temperature - args.min_temperature) * Prob_OOV
        else:
            new_temperature = args.min_temperature
        # print('New temperature = ', format(new_temperature, '.4f'))
        softmax_filtered_probabilities = torch.nn.functional.softmax(torch.tensor(filtered_probabilities) / new_temperature, dim = -1)

        new_dict = dict()
        for i in range(len(softmax_filtered_probabilities)):
            new_dict[indices[i]] = Decimal(softmax_filtered_probabilities[i].item())

        # print('Candidate size = ',len(new_dict))

        if len(new_dict) > 1:
            if lexicon_flag == True:
                new_dict = lexicon(new_dict, current_node)
            next_token, current_min, current_max = generate_next_token(current_min, current_max, new_dict, secret_decimal)
        else:
            next_token = indices[0]



        next_char = tokenizer.decode([next_token]).lower()
        if not next_char.isalpha():
            current_node = trie.root
            Word_num += 1
            lexicon_flag = True
        elif next_char in current_node.children and lexicon_flag == True:
            current_node = current_node.children[next_char]
        elif next_char not in current_node.children and lexicon_flag == True:
            lexicon_flag = False
            OOV_num += 1
            print('OOV!')
        else:
            lexicon_flag = False


        # token_with_max_probability = max(token_probability_dict, key = lambda x: token_probability_dict[x])
        # print(token_with_max_probability)
        # print(tokenizer.decode([torch.tensor(token_with_max_probability)]))
        # print(next_token)
        input_ids = torch.cat((input_ids,torch.tensor(next_token).unsqueeze(0).unsqueeze(0)), dim=1)
        Avg_entropy += Entropy
        Entropy_list.append(round(float(Entropy), 4))
        num += 1

        print(tokenizer.decode(input_ids[0][-num:]))
        print(format(secret_decimal - current_min, '.2f'))
        print()
        if secret_decimal-current_min < 1.0:
            break
    Avg_entropy /= num
    # print('-----------------------------------------------------------------------------------------------------------------------------------------------------------------')
    # print(Entropy_list)
    # print('Average entropy = ', format(Avg_entropy, '.4f'))
    # print('Bits per character = ', format(args.secret_length/num, '.4f'))
    # print('OOV number = ', format(OOV_num))
    return input_ids[0][-num:], OOV_num/Word_num

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    # parser.add_argument('--top_k', default=32, type=int, required=False)
    # parser.add_argument('--min_p', default=0.01, type=float, required=False)
    parser.add_argument('--max_temperature', default=2.00, type=float, required=False)
    parser.add_argument('--min_temperature', default=2.00, type=float, required=False)
    parser.add_argument('--secret_length', default=128, type=int, required=False)
    parser.add_argument('--sigma_multiple', default=0.0, type=float, required=False)
    parser.add_argument('--OOV_multiple', default=1.0, type=float, required=False)
    # parser.add_argument('--seed', default=3223, type=int, required=False)
    parser.add_argument('--generation_num', default=1, type=int, required=False)
    args = parser.parse_args()

    for _ in range(args.generation_num):
        input_sentence = 'Paris is the capital of France. '
        input_ids = tokenizer.encode(input_sentence, return_tensors="pt")
        secret_bits = np.random.randint(0, 2, args.secret_length)
        secret_str = ''.join([str(i) for i in secret_bits.tolist()])
        secret_decimal = Decimal(0)
        for i in range(args.secret_length):
            secret_decimal += Decimal(Decimal(2 ** (args.secret_length - i - 1)) * int(secret_bits[i]))

        output_ids, OOV_rate = generate(input_ids, secret_decimal)
        stega_text = tokenizer.decode(output_ids)
        print(stega_text)
