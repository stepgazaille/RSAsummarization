import subprocess
import sys
from os import path, listdir, makedirs, remove
from shutil import rmtree
from bs4 import BeautifulSoup


dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence


source_path = './duc2007_testdocs/main/'
tmp_path = '/tmp/'
tmp_texts_path = path.join(tmp_path, 'duc_docs/')
tmp_tokens_path = path.join(tmp_path, 'duc_docs_tokens/')
dest_path = './duc07tokenized/testdata/docs/'
map_file = path.join(tmp_path, 'mapping.txt') 


for topic in listdir(source_path):
    topic_path = path.join(source_path, topic)
    makedirs(path.join(tmp_texts_path, topic))
    makedirs(path.join(tmp_tokens_path, topic))

    with open(map_file, "w") as f:
        for doc_name in listdir(topic_path):
            file_path = path.join(topic_path, doc_name)
            with open(file_path) as doc_file:
                soup = BeautifulSoup(doc_file, features='html.parser')
                for doc in soup('doc'):
                    doc_id = doc.findChild('docno').text
                    if doc.findChild('text') is not None:
                        text = doc.findChild('text').text
                    else:
                        text = doc.findChild('graphic').text

                    text_file = path.join(tmp_texts_path, topic, doc_name)
                    token_file = path.join(tmp_tokens_path, topic, doc_name)
                    with open(text_file,'w') as tmp_file:
                        tmp_file.write(text)
                        tmp_file.close()

                f.write("%s \t %s\n" % (text_file, token_file))

    command = ['java', 'edu.stanford.nlp.process.PTBTokenizer', '-ioFileList', '-preserveLines', map_file]
    print("Tokenizing %i files in %s and saving in %s..." % (len(listdir(topic_path)), topic_path, path.join(tmp_tokens_path, topic)))
    subprocess.call(command)
    remove(map_file)

rmtree(tmp_texts_path)
print("Stanford CoreNLP Tokenizer has finished.")


for topic in listdir(tmp_tokens_path):

    topic_path = path.join(tmp_tokens_path, topic)
    topic_dest = path.join(dest_path, topic)
    makedirs(topic_dest)
    
    print("Preprocessing %i files in %s and saving in %s..." % (len(listdir(topic_path)), topic_path, topic_dest))
    for doc_name in listdir(topic_path):

        token_file = path.join(topic_path, doc_name)
        file_dest = path.join(topic_dest, doc_name)
        
        with open(token_file, 'r') as t_file:
            text = t_file.read()

        # Lowercase everything
        text = text.lower()
            
        with open(file_dest,'w') as d_file:
            d_file.write(text)

rmtree(tmp_tokens_path)

