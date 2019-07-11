import os
import numpy as np
import sentencepiece as spm
from tqdm import tqdm
from nltk import tokenize
import json

class Subwords():
	"""
		Subword implementation supporting BPE algorith. 
		Also containing preprocessing and training vector utilities.
			
		Attributes: VOCAB_SIZE  number of subwords in the vocab
	"""
	
	VOCAB_SIZE = 200

	def __init__(self, do_lowercase = True, squad_version = "1.1"):
		"""
		Loads context dataset and trains the subword model.
		"""
		
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
			print("Train Vocab with BPE...")
			spm.SentencePieceTrainer.Train('--input={} --model_prefix={} --vocab_size={} --model_type=bpe --normalization_rule_name={}'.format(
					context_file, os.path.join(base_dir,'subword'), self.VOCAB_SIZE, normalization))
			print("Train Vocab with BPE done.")
			model = sp.Load(os.path.join(base_dir,'subword.model'))

		self.vectors = sp
	
	def getContextData(self,train_file, dest_path):
		"""
		Returns the context data of the specified data file. 
		The input data is split by sentences.
			
		:param: train_file 	path to json dataset file
				dest_path	path for saving the processed dataset
		"""
		
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
		"""
		Computes normalized vector representation of subwords. 
		Appends them to the respective input embedding batch. 
		
		:param: batch	input embedding batch
				input	list of tokensized words 
		"""
		
		new_batch = np.zeros(shape=(batch.shape[:-1] + (batch.shape[-1] + len(self.vectors),)), dtype='float32')
		for i in range(len(input)):
		    for j in range(len(input[i])):
		        ids = self.vectors.EncodeAsIds(input[i][j])
		        encoded_ids = np.zeros(shape=len(self.vectors), dtype='float32')
		        for id in ids:
		            encoded_ids[id] = 1 / len(ids)
		        new_batch[i][j] = np.concatenate((batch[i][j], encoded_ids))

		return new_batch
