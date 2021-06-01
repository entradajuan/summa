!pip install transformers==4.5.1
!pip install torch==1.8.1
!pip install sentencepiece==0.1.94

display_architecture=True

import torch
import json 
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

model = T5ForConditionalGeneration.from_pretrained('t5-large')
tokenizer = T5Tokenizer.from_pretrained('t5-large')
device = torch.device('cpu')
