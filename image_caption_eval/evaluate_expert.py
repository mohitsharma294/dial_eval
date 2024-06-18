import json
import torch
from transformers import AutoProcessor, AutoTokenizer, CLIPModel, RobertaForMaskedLM
from PIL import Image
import os
from nltk.corpus import stopwords
import random
import yake
import spacy
import collections
import argparse
from scipy.stats import kendalltau, pearsonr, spearmanr
import numpy as np
from tqdm import tqdm
import math

parser = argparse.ArgumentParser()
parser.add_argument('-path','--path', help='path of the model directiory', required=True)
parser.add_argument('-out','--out', help='name of output directory', required=True)
parser.add_argument('-lbl','--lbl', help='file label', required=False, default="")
parser.add_argument('-size','--size', help='Model size', default='base', choices=['base', 'large'])
parser.add_argument('-keys','--keys', help='Number of keywords', type=int, required=False, default=20)
parser.add_argument('-no_pos','--no_pos', help='Skip POS keywords', default=True, action='store_false')
parser.add_argument('-no_pre','--no_pre', help='Inference for model with no pre-training', default=False, action='store_true')
parser.add_argument('-r','--r', help='Percentage masking', type=float, required=False, default=0.5)
#parser.add_argument('-test','--test', help='Test Run', default=False, action='store_true')
args = vars(parser.parse_args())
model_dir = args['path']
numOfKeywords = args['keys']
add_pos = args['no_pos']
out_label = args['lbl']
no_pre = args['no_pre']
model_size = args['size']
m_ratio = args['r']

if(model_dir=="roberta-base"):
    print("Evaluating with roberta-base model.")
    model_dir = "roberta-base"
    out_dir = "result_roberta_base"
    no_pre = True
else:
    if(not os.path.isdir(model_dir)):
        print("Model Directory does not exist.")
        exit(0)
    else:
        out_dir = os.path.join(model_dir, args['out'])
    
if(not os.path.isdir(out_dir)):
    os.mkdir(out_dir)

print(f"Model directory: {model_dir}")
print(f"Output directory: {out_dir}")
print(f"numOfKeywords: {numOfKeywords}")
print(f"no_pre: {no_pre}")
print(f"m_ratio: {m_ratio}")

#----------------------------

SEED = 10
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)      
    device = torch.device("cuda")
    print(f"Using GPUs!")
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

IMAGE_MODEL = "openai/clip-vit-base-patch32"
ROBERTA_MODEL = "roberta-base"
if(model_size=="large"):
    IMAGE_MODEL = "openai/clip-vit-large-patch14"
    ROBERTA_MODEL = "roberta-large"
    
lst_pos_tags = ['NN', 'NNP', 'NNS', 'JJ', 'CD', 'VB', 'VBN', 'VBD', 'VBG', 'RB', 'VBP', 'VBZ', 'NNPS', 'JJS']
stop_words = stopwords.words('english')
base_path = "evaluation/flickr8k"

#----------------------------

top_n = numOfKeywords
language = "en"
max_ngram_size = 1
deduplication_threshold = 0.9
deduplication_algo = 'seqm'
windowSize = 1
custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=top_n, features=None)

nlp = spacy.load("en_core_web_sm")
mask_random = False
max_keyword = numOfKeywords
img_len = 50
if(model_size=="large"):
    img_len = 257

processor = AutoProcessor.from_pretrained(IMAGE_MODEL)
tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL)
clip_model = CLIPModel.from_pretrained(IMAGE_MODEL)
model = RobertaForMaskedLM.from_pretrained(model_dir)
clip_model.to(device)
model.to(device)
clip_model.eval()
model.eval()
print("Model loaded")

#----------------------------

def tokenize_sentence(txt, tokenizer):
    #result = tokenizer(txt, max_length=max_len_caption, truncation=True)
    result = tokenizer(txt)
    word_ids = result.word_ids()
    if tokenizer.is_fast:
        result["word_ids"] = [word_ids[i] for i in range(len(result["input_ids"]))]
    return result

def get_word_mapping(tok):
    word_ids = tok["word_ids"].copy()
    mapping = collections.defaultdict(list)
    current_word_index = -1
    current_word = None
    for idx, word_id in enumerate(word_ids):
        if word_id is not None:
            if word_id != current_word:
                current_word = word_id
                current_word_index += 1
            mapping[current_word_index].append(idx)
    return mapping

def get_pos_tags(doc):
    pos_tags = {}
    for token in doc:
        if(not (token.is_stop or token.is_punct or token.is_space or token.text.lower() in stop_words)):
            if(token.tag_ in lst_pos_tags):
                pos_tags[token.text] = token.tag_
    return pos_tags

def get_mask_words(txt, tok, mapping, add_pos):
    if(mask_random):
        n_sample = math.ceil(0.15*len(mapping))
        mask = random.sample(range(len(mapping)),n_sample)
        mask_words = []
        for idx in mask:
            start, end = tok.word_to_chars(idx)
            word = txt[start:end].lower()
            mask_words.append(word)
    else:
        yake_doc = txt.replace(tokenizer.eos_token, "")
        yake_doc = yake_doc.replace(tokenizer.bos_token, "")
        yake_doc = yake_doc.strip()
        
        #max_keyword = 3
        #max_keyword = max(3, math.ceil(0.15*len(mapping)))
        max_keyword = max(3, math.ceil(m_ratio*len(mapping)))
        keywords = custom_kw_extractor.extract_keywords(yake_doc)[:max_keyword]
        #keywords = custom_kw_extractor.extract_keywords(yake_doc)
        
        lst_kw = [kw[0].lower() for kw in keywords]
        
        if(len(lst_kw)<max_keyword and add_pos):
            n = max_keyword-len(lst_kw)
            txt_doc = nlp(txt)
            pos_tags = get_pos_tags(txt_doc)
            for w in pos_tags:
                if(w not in lst_kw):
                    lst_kw.append(w.lower())
                    n = n-1
                    if(n==0):
                        break

        mask = []
        mask_words = []
        for idx in mapping:
            start, end = tok.word_to_chars(idx)
            word = txt[start:end].lower()
            if word in lst_kw:
                mask.append(idx)
                mask_words.append(word)
    return mask, mask_words

def get_masked_tokens(tokenizer, tok, mapping, mask):
    input_ids = tok["input_ids"].copy()
    labels = [-100]*len(input_ids)
    for word_id in mask:
        for idx in mapping[word_id]:
            labels[idx] = input_ids[idx]
            input_ids[idx] = tokenizer.mask_token_id
    return input_ids, labels

def evaluate(encoder_hidden_states, input_id, lbl, attn_mask):
    b_input_ids = torch.tensor([input_id], dtype=torch.long).to(device)
    b_labels = torch.tensor([lbl], dtype=torch.long).to(device)
    b_attn_mask = torch.tensor([attn_mask], dtype=torch.long).to(device)

    with torch.no_grad():
        inputs_embeds = model.roberta.embeddings.word_embeddings(b_input_ids)
        f_inputs_embeds = torch.cat((encoder_hidden_states, inputs_embeds), dim=1)
        output = model(inputs_embeds=f_inputs_embeds, attention_mask=b_attn_mask, labels=b_labels)
        loss = output.loss.item()
    return loss

def get_score(caption, encoder_hidden_states):
    tok_caption = tokenize_sentence(caption, tokenizer)
    map_caption = get_word_mapping(tok_caption)
    mask, mask_words = get_mask_words(caption, tok_caption, map_caption, True)
    tok_masked, label = get_masked_tokens(tokenizer, tok_caption, map_caption, mask)
    
    attn_mask = [1]*(len(tok_masked) + img_len)
    f_lbl = [-100]*img_len
    f_lbl.extend(label)
    
    logger.write(f"mask_words: {mask_words}\n")
    logger.write(f"mask: {mask}\n")
    
    score = evaluate(encoder_hidden_states, tok_masked, f_lbl, attn_mask)
    return round(score, 4)

#----------------------------

dataset = "flickr8k-expert"
data_path = "evaluation/flickr8k/flickr8k.json"

#dataset = "flickr8k-cf"
#data_path = "evaluation/flickr8k/crowdflower_flickr8k.json"

with open(data_path, "r") as f:
    data1 = json.load(f)
print(len(data1))

out_path = os.path.join(out_dir, f'out_{dataset}{out_label}.txt')
logger = open(out_path, "w")

lst_gt = []
lst_pr = []

#for k in data1:
count = 0
for k in tqdm(data1):
    v = data1[k]
    image_path = os.path.join(base_path, v['image_path'])
    human_judgement = v["human_judgement"]
    ground_truth = v["ground_truth"]
    logger.write("-"*30+"\n")
    logger.write(f"id: {k}\n")
    logger.write(f"image_path: {image_path}\n")
    logger.write(f"ground_truth: {ground_truth}\n")
    
    img = Image.open(image_path)
    #img.show()
    pixel_values = processor(images=img, return_tensors="pt")["pixel_values"].to(device)
    with torch.no_grad():
        vision_outputs = clip_model.vision_model(pixel_values, output_attentions=False, 
                                                     output_hidden_states=False)
        encoder_hidden_states = vision_outputs.last_hidden_state
        logger.write(f"encoder_hidden_states: {encoder_hidden_states.shape}\n")
    
    logger.write("-"*20+"\n")
    for i in range(len(human_judgement)):
        if(i%3!=0):
            continue
        caption = human_judgement[i]["caption"]
        #caption = human_judgement[i]["caption"].lower()
        rating = human_judgement[i]["rating"]
        logger.write(f"caption: {caption}\n")
        logger.write(f"rating: {rating}\n")
        lst_gt.append(rating)
        #lst_gt.append(rating)
        #lst_gt.append(rating)
        
        score = get_score(caption.strip(), encoder_hidden_states)
        logger.write(f"score: {score}\n")
        lst_pr.append(score)
        #lst_pr.append(score)
        #lst_pr.append(score)
        
        logger.write("-"*20+"\n")
        
    logger.write("-"*20+"\n")
    logger.write("Groundtruths:-\n")
    for gt in ground_truth:
        score = get_score(gt.strip(), encoder_hidden_states)
        logger.write(f"gt: {gt}\n")
        logger.write(f"score: {score}\n")
        logger.write("-"*20+"\n")
    
    count+=1
    #if(count==20):
    #    break
    
logger.write("="*20+"\n")
x = np.array(lst_gt)
y = np.array(lst_pr)
logger.write(f"X: {len(x)}\n")
logger.write(f"Y: {len(y)}\n")

score_file = os.path.join(out_dir, f'scores_{dataset}{out_label}.txt')
lst_scores = []
for i in range(len(x)):
    lst_scores.append((x[i], y[i]))
# Serializing json
json_object = json.dumps(lst_scores, indent=4) 
# Writing to sample.json
with open(score_file, "w") as outfile:
    outfile.write(json_object)

tau, pval = kendalltau(x, y, variant='c')
logger.write(f"Kendall tau-c: ({tau}, {pval})\n")

tau, pval = kendalltau(x, y, variant='c', alternative='less')
logger.write(f"Kendall tau-c less: ({tau}, {pval})\n")

tau, pval = kendalltau(x, y, variant='b')
logger.write(f"Kendall tau-b: ({tau}, {pval})\n")

tau, pval = kendalltau(x, y, variant='b', alternative='less')
logger.write(f"Kendall tau-b less: ({tau}, {pval})\n")

corr1, p_val1 = pearsonr(x, y)
corr2, p_val2 = spearmanr(x, y)
logger.write(f"Correlation between GT and Score: Pearson = ({round(corr1,4)}, {round(p_val1,4)}), Spearman = ({round(corr2,4)}, {round(p_val2,4)})\n")

logger.close()

print("done")