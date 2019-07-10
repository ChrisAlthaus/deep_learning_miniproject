import os
import numpy as np
import sentencepiece as spm
from tqdm import tqdm
from nltk import tokenize
import json

class Subwords():

	def __init__(self, emdim, do_lowercase = False, squad_version = "1.1"):
		
		base_dir = os.path.relpath(os.path.join(os.path.dirname(__file__), os.pardir, 'data', 'sentencepiece'))
		data_dir = os.path.relpath(os.path.join(os.path.dirname(__file__), os.pardir, 'data'))
		
		
		sp = spm.SentencePieceProcessor()
		model = None
		try:
			model = sp.Load(os.path.join(base_dir,'subword.model'))
		except:
			train_filename = "train-v{}.json".format(squad_version)
			train_file = os.path.join(data_dir,'squad', train_filename)
			
			if not os.path.exists(base_dir):
				os.makedirs(base_dir)
			
			context_file = self.getContextData(train_file, base_dir)
			
			if do_lowercase:
				normalization = "nmt_nfkc_cf"
			else:
				normalization = "nmt_nfkc"

			# --character_coverage=1.0
			spm.SentencePieceTrainer.Train('--input={} --model_prefix={} --vocab_size={} --model_type=bpe --normalization_rule_name={}'.format(
					context_file, os.path.join(base_dir,'subword'), emdim, normalization))
			model = sp.Load(os.path.join(base_dir,'subword.model'))

		self.vectors = sp
	
	def getContextData(self,train_file, dest_path):
		dataset = None
		
		with open(train_file) as data_file:
			dataset = json.load(data_file)
		
		file_name = os.path.join(dest_path, '{}'.format("context.data"))
			
		with open(file_name, 'w', encoding='utf-8') as context_file:
			
			for articles_id in range(len(dataset['data'])):
				article_paragraphs = dataset['data'][articles_id]['paragraphs']
				for pid in range(len(article_paragraphs)):

					context = article_paragraphs[pid]['context'].strip()  # string

					# The following replacements are suggested in the paper
					# BidAF (Seo et al., 2016)
					context = context.replace("''", '" ')
					context = context.replace("``", '" ')
					
					sentences = tokenize.sent_tokenize(context)
					
					for s in sentences:
						context_file.write(s + '\n')
		
		return file_name
		

	def append_batch(self, batch, input):
		new_batch = np.zeros(shape=(batch.shape[:-1] + (batch.shape[-1] + len(self.vectors),)))
		for i in range(len(input)):
		    for j in range(len(input[i])):
		        ids = self.vectors.EncodeAsIds(input[i][j])
		        encoded_ids = np.zeros(shape=len(self.vectors), dtype='float32')
		        for id in ids:
		            encoded_ids[id] = 1 / len(ids)
		        new_batch[i][j] = np.concatenate((batch[i][j], encoded_ids))

		return batch

		
		
"""		
	 train_filename = "train-v{}.json".format(squad_version)
    dev_filename = "dev-v{}.json".format(squad_version)

    # download train set
    maybe_download(SQUAD_BASE_URL, train_filename, data_dir)

    # read train set
    train_data = data_from_json(os.path.join(data_dir, train_filename))
    print("Train data has %i examples total" % total_examples(train_data))	
		

>>> 
>>> 
True
>>> sp.EncodeAsPieces("This is a test")
['\xe2\x96\x81This', '\xe2\x96\x81is', '\xe2\x96\x81a', '\xe2\x96\x81', 't', 'est']
>>> sp.EncodeAsIds("This is a test")
[284, 47, 11, 4, 15, 400]
>>> sp.DecodePieces(['\xe2\x96\x81This', '\xe2\x96\x81is', '\xe2\x96\x81a', '\xe2\x96\x81', 't', 'est'])
'This is a test'
>>> sp.NBestEncodeAsPieces("This is a test", 5)
[['\xe2\x96\x81This', '\xe2\x96\x81is', '\xe2\x96\x81a', '\xe2\x96\x81', 't', 'est'], ['\xe2\x96\x81This', '\xe2\x96\x81is', '\xe2\x96\x81a', '\xe2\x96\x81', 'te', 'st'], ['\xe2\x96\x81This', '\xe2\x96\x81is', '\xe2\x96\x81a', '\xe2\x96\x81', 'te', 's', 't'], ['\xe2\x96\x81This', '\xe2\x96\x81is', '\xe2\x96\x81a', '\xe2\x96\x81', 't', 'e', 'st'], ['\xe2\x96\x81This', '\xe2\x96\x81is', '\xe2\x96\x81a', '\xe2\x96\x81', 't', 'es', 't']]
>>> for x in range(10):
...     sp.SampleEncodeAsPieces("This is a test", -1, 0.1)
...
['\xe2\x96\x81', 'T', 'h', 'i', 's', '\xe2\x96\x81', 'is', '\xe2\x96\x81a', '\xe2\x96\x81', 't', 'e', 's', 't']
['\xe2\x96\x81T', 'h', 'is', '\xe2\x96\x81is', '\xe2\x96\x81', 'a', '\xe2\x96\x81', 't', 'est']
['\xe2\x96\x81This', '\xe2\x96\x81is', '\xe2\x96\x81', 'a', '\xe2\x96\x81', 't', 'e', 'st']
['\xe2\x96\x81This', '\xe2\x96\x81is', '\xe2\x96\x81a', '\xe2\x96\x81', 't', 'e', 'st']
['\xe2\x96\x81This', '\xe2\x96\x81is', '\xe2\x96\x81a', '\xe2\x96\x81', 't', 'e', 's', 't']
['\xe2\x96\x81T', 'h', 'is', '\xe2\x96\x81', 'i', 's', '\xe2\x96\x81a', '\xe2\x96\x81', 'te', 's', 't']
['\xe2\x96\x81This', '\xe2\x96\x81', 'is', '\xe2\x96\x81a', '\xe2\x96\x81', 'te', 's', 't']
['\xe2\x96\x81This', '\xe2\x96\x81', 'i', 's', '\xe2\x96\x81a', '\xe2\x96\x81', 't', 'e', 'st']
['\xe2\x96\x81This', '\xe2\x96\x81', 'is', '\xe2\x96\x81', 'a', '\xe2\x96\x81', 't', 'e', 'st']
['\xe2\x96\x81This', '\xe2\x96\x81', 'i', 's', '\xe2\x96\x81', 'a', '\xe2\x96\x81', 'te', 's', 't']
>>> sp.DecodeIds([284, 47, 11, 4, 15, 400])
'This is a test'
>>> sp.GetPieceSize()
1000
>>> sp.IdToPiece(2)
'</s>'
>>> sp.PieceToId('</s>')
2
>>> len(sp)
1000
>>> sp['</s>']"""