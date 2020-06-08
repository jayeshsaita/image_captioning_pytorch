from torchvision import models as tvm
import torch
import json
import torch
import torch.nn.functional as F
import argparse
import src.model as model
import cv2
import matplotlib.pyplot as plt
import numpy as np


def predict_caption(img_feature, ic_model):
    sentence = []

    word2idx = json.load(open('data/word2idx.json'))
    idx2word = json.load(open('data/idx2word.json'))

    ic_model.eval()
    img_feature = ic_model.img_hidden(img_feature)
    img_feature = ic_model.img_bn(img_feature)
    img_feature = F.relu(img_feature)
  
    img_embed = ic_model.img_embedding(img_feature)
    out, (hn, cn) = ic_model.lstm(img_embed.view(1,1,-1))
    fc1 = ic_model.fc1(hn[-1])
    fc1 = ic_model.fc1_dropout(fc1)
    fc1 = F.relu(ic_model.fc1_bn(fc1))
    fc2 = ic_model.fc2(fc1)
    fc2 = torch.softmax(fc2, dim=1)
    pw = fc2.argmax()
    if pw.item() != 8186: # <start> word
        sentence.append(idx2word[str(pw.item())])

    for _ in range(85):
        em = ic_model.vocab_embedding(pw)
        out, (hn, cn) = ic_model.lstm(em.view(1,1,-1), (hn, cn))
        fc1 = ic_model.fc1(hn[-1])
        fc1 = ic_model.fc1_dropout(fc1)
        fc1 = F.relu(ic_model.fc1_bn(fc1))
        fc2 = ic_model.fc2(fc1)
        fc2 = torch.softmax(fc2, dim=1)
        pw = fc2.argmax()
    
        if pw.item() == 8187: # <end> word
            break
    
        sentence.append(idx2word[str(pw.item())])

    return ' '.join(sentence)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Script to generate caption on a given image.')
    ap.add_argument('-m', '--model', required=True, help='Path to model checkpoint')
    ap.add_argument('-p', '--path', required=True, help='Path to image')
    ap.add_argument('-s', '--save', action='store_true', help='Save output image with caption')
    args = vars(ap.parse_args())

    print('Loading backbone')
    rn50 = tvm.resnet50(pretrained=True)
    rn50 = torch.nn.Sequential(*list(rn50.children())[:-1])
    rn50.eval()

    print('Loading model')
    ic_model = model.CNNLSTM.load_from_checkpoint(args['model'])

    print('Loading Image')
    img = cv2.imread(args['path'])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = torch.from_numpy(img)
    img = img.to(dtype=torch.float32)
    img /= 255
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(3,1,1)
    img = img.permute(2,0,1)
    img = (img - mean)/std

    print('Generating Caption')
    ic_model.eval()
    ic_model.freeze()
    with torch.no_grad():
        img_feature = rn50(img.unsqueeze(0))
    caption = predict_caption(img_feature.squeeze().unsqueeze(0), ic_model)
    print('*'*40 + 'Generated Caption' + '*'*40)
    print(caption)
    print('*'*97)

    if args['save']:
        img = cv2.imread(args['path'])
        img_name = args['path'].split('/')[-1]
        h, w, _ = img.shape
        img_white = np.ones((h+200, w+200, 3)) * 255
        h, w, _ = img_white.shape
        img_white[100:-100, 100:-100] = img
        caption = caption.split(' ')
        caption_1 = caption[:len(caption)//2]
        caption_2 = caption[len(caption)//2:]
        cv2.putText(img_white, ' '.join(caption_1), (100, h-70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), thickness=2)
        cv2.putText(img_white, ' '.join(caption_2), (100, h-40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), thickness=2)
        cv2.imwrite(f'output_{img_name}', img_white)
