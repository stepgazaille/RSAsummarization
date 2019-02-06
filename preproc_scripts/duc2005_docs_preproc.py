import subprocess
import pathlib
import sys
from pathlib import Path
from os import listdir, remove
from shutil import rmtree
from bs4 import BeautifulSoup


source_path = Path('./duc2005_docs/')
dest_path = Path('./duc05tokenized/testdata/docs/')
tmp_path = Path('/tmp/')
map_file = 'mapping.txt'


for topic_path in listdir(source_path):

    with open(tmp_path/map_file, "w") as f:
        for file_path in listdir(source_path/topic_path):

            with open(source_path/topic_path/file_path) as doc_file:
                soup = BeautifulSoup(doc_file, features='html.parser')
                for doc in soup('doc'):
                    doc_id = doc.findChild('docno').text
                    if doc.findChild('text') is not None:
                        text = doc.findChild('text').text
                    else:
                        text = doc.findChild('graphic').text

                    pathlib.Path(tmp_path/topic_path).mkdir(parents=True, exist_ok=True)
                    pathlib.Path(dest_path/topic_path).mkdir(parents=True, exist_ok=True) 
                    with open(tmp_path/topic_path/file_path,'w') as tmp_file:
                        tmp_file.write(text)
                        tmp_file.close()

                f.write("%s \t %s\n" % (tmp_path/topic_path/file_path, dest_path/topic_path/file_path))
        f.close()

        command = ['java', 'edu.stanford.nlp.process.PTBTokenizer', '-ioFileList', '-preserveLines', tmp_path/map_file]
        print("Tokenizing %i files in %s and saving in %s..." % (len(listdir(source_path/topic_path)), source_path/topic_path, dest_path/topic_path))
        subprocess.call(command)
        remove(tmp_path/map_file)
        rmtree(tmp_path/topic_path)

print("Stanford CoreNLP Tokenizer has finished.")
