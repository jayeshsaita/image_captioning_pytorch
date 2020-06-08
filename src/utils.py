from nltk.translate import bleu_score
from nltk.translate.bleu_score import sentence_bleu
import torch
import json


word2idx = json.load(open('data/word2idx.json'))
idx2word = json.load(open('data/idx2word.json'))

def pad_seq_to_caption(pad_seq):
    sentence = []
    for seq in pad_seq:
        if seq == 3002:
            continue
        if seq == 3003:
            break
        if seq == 0:
            sentence.append('<pad>')
        else:
            sentence.append(idx2word[str(seq.item())])
  
    return ' '.join(sentence)


def bleu_4(input, selected_caption, all_captions):
    bleu_scores = []

    input = torch.softmax(input, dim=1)
    input = input.view(-1, 85, 8189)
    for inp, tar_all_captions in zip(input, all_captions):
        inp = torch.argmax(inp, dim=1)
        predicted_sentence = pad_seq_to_caption(inp).split()
        references = []
        for ref in tar_all_captions:
            references.append(pad_seq_to_caption(ref).split())
    
        bleu_scores.append(sentence_bleu(references, predicted_sentence, weights=(0.25, 0.25, 0.25, 0.25)))
  
    bleu_scores = torch.Tensor(bleu_scores)
    return bleu_scores.mean()

