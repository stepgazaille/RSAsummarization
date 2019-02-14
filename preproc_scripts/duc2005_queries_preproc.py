import subprocess
import sys
from os import path, listdir, makedirs, remove
from shutil import rmtree
from bs4 import BeautifulSoup


dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence


source_file = 'duc2005_topics.sgml'
tmp_path = '/tmp/'
tmp_texts_path = path.join(tmp_path, 'duc_queries/')
tmp_tokens_path = path.join(tmp_path, 'duc_queries_tokens/')
dest_path = './duc05tokenized/queries/'
map_file = path.join(tmp_path, 'mapping.txt') 


nb_queries = 0
makedirs(tmp_texts_path)
makedirs(tmp_tokens_path)
with open(map_file, "w") as f:
    with open(source_file) as topics_file:

        soup = BeautifulSoup(topics_file, features='html.parser')
        for topic in soup('topic'):
            nb_queries += 1
            topic_id = topic.findChild('num').text.strip()
            title = topic.findChild('title').text
            query = topic.findChild('narr').text

            text_file = path.join(tmp_texts_path, topic_id)
            token_file = path.join(tmp_tokens_path, topic_id)
            with open(text_file, 'w') as query_file:
                query_file.write(query)
                query_file.close()
            f.write("%s \t %s\n" % (text_file, token_file))

command = ['java', 'edu.stanford.nlp.process.PTBTokenizer', '-ioFileList', '-preserveLines', map_file]
print("Tokenizing %i queries in %s and saving in %s..." % (nb_queries, source_file, dest_path))
subprocess.call(command)
print("Stanford CoreNLP Tokenizer has finished.")
remove(map_file)
rmtree(tmp_texts_path)


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

