import subprocess
import pathlib
from pathlib import Path
from os import listdir, remove


source_path = Path('./models/')
dest_path = Path('./duc05tokenized/models/')
tmp_path = Path('/tmp/')
map_file = 'mapping.txt'

with open(tmp_path/map_file, "w") as f:
    for file_path in listdir(source_path):
        f.write("%s \t %s\n" % (source_path/file_path, dest_path/file_path))
    f.close()

pathlib.Path(dest_path).mkdir(parents=True, exist_ok=True) 
command = ['java', 'edu.stanford.nlp.process.PTBTokenizer', '-ioFileList', '-preserveLines', tmp_path/map_file]
print("Tokenizing %i files in %s and saving in %s..." % (len(listdir(source_path)), source_path, dest_path))
subprocess.call(command)
remove(tmp_path/map_file)
print("Stanford CoreNLP Tokenizer has finished.")
