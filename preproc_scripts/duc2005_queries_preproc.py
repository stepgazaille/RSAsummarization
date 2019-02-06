import subprocess
import pathlib
from pathlib import Path
from bs4 import BeautifulSoup
from shutil import rmtree
from os import remove

source_file = 'duc2005_topics.sgml'
tmp_path = Path('/tmp/')
map_file = 'mapping.txt'
dest_path = Path('./duc05tokenized/queries/')


nb_queries = 0
pathlib.Path(dest_path).mkdir(parents=True, exist_ok=True) 
pathlib.Path(tmp_path/'queries').mkdir(parents=True, exist_ok=True) 
with open(tmp_path/map_file, "w") as f:

    with open(source_file) as topics_file:

        soup = BeautifulSoup(topics_file, features='html.parser')
        for topic in soup('topic'):
            nb_queries += 1
            topic_id = topic.findChild('num').text.strip()
            title = topic.findChild('title').text
            query = topic.findChild('narr').text
            
            with open(tmp_path/'queries'/topic_id, 'w') as query_file:
                query_file.write(query)
                query_file.close()
            
            f.write("%s \t %s\n" % (tmp_path/'queries'/topic_id, dest_path/topic_id))
f.close()


command = ['java', 'edu.stanford.nlp.process.PTBTokenizer', '-ioFileList', '-preserveLines', tmp_path/map_file]
print("Tokenizing %i files in %s and saving in %s..." % (nb_queries, source_file, dest_path))
subprocess.call(command)
remove(tmp_path/map_file)
rmtree(tmp_path/'queries')
print("Stanford CoreNLP Tokenizer has finished.")
