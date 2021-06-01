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

def summarize(text,ml):
  preprocess_text = text.strip().replace("\n","")
  t5_prepared_Text = "summarize: "+preprocess_text
  print ("Preprocessed and prepared text: \n", t5_prepared_Text)

  tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)

  # summmarize 
  summary_ids = model.generate(tokenized_text,
                                      num_beams=4,
                                      no_repeat_ngram_size=2,
                                      min_length=30,
                                      max_length=ml,
                                      early_stopping=True)

  output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
  return output

text="""
Brand new/unworn Rolex GMT-Master II 126710BLNR 'Batman', complete with box & papers, dated 2021.

The photos displayed are from the actual watch for sale.

EU/worldwide shipping on request. Shipping within Greece is free of charge.

Feel free to contact us for any further information.

Please note the following:

- Price is excluding shipping.
- We do not sell tax-free.
- For watches marked as "available now", we normally ship within 1 business day upon receiving payment.
- We sell only 100% authentic watches, all watches displayed are fully checked for authenticity.
- All new/unworn watches are accompanied by their genuine documents, including a stamped and dated international manufacturer's warranty.
- As a company, we do not have a physical store. For security reasons, we do not keep our stock at our offices. Watch viewings, whenever possible, are strictly by appointment only.
"""
print("Number of characters:",len(text))
summary=summarize(text,50)
print ("\n\nSummarized text: \n",summary)
