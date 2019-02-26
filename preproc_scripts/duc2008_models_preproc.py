import subprocess
import sys
from os import path, listdir, makedirs, remove
from shutil import rmtree


dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence


source_path = './models/'
tmp_path = '/tmp/'
tmp_tokens_path = path.join(tmp_path, 'duc_models_tokens/')
dest_path = './duc08tokenized/models/'
map_file = path.join(tmp_path, 'mapping.txt') 


makedirs(tmp_tokens_path)
with open(map_file, "w") as f:
    for doc_name in listdir(source_path):       
        file_path = path.join(source_path, doc_name)
        token_file = path.join(tmp_tokens_path, doc_name)
        f.write("%s \t %s\n" % (file_path, token_file))

command = ['java', 'edu.stanford.nlp.process.PTBTokenizer', '-ioFileList', '-preserveLines', map_file]
print("Tokenizing %i files in %s and saving in %s..." % (len(listdir(source_path)), source_path, tmp_tokens_path))
subprocess.call(command)
remove(map_file)
print("Stanford CoreNLP Tokenizer has finished.")


makedirs(dest_path)
print("Preprocessing %i files in %s and saving in %s..." % (len(listdir(tmp_tokens_path)), tmp_tokens_path, dest_path))
for doc_name in listdir(tmp_tokens_path):
    token_file = path.join(tmp_tokens_path, doc_name)
    file_dest = path.join(dest_path, doc_name)

    with open(token_file, 'r') as t_file:
        text = t_file.read()
    
    # Lowercase everything
    text = text.lower()

    with open(file_dest,'w') as d_file:
        d_file.write(text)

rmtree(tmp_tokens_path)

