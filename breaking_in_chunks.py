import json
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
input='/content/drive/MyDrive/input/all_pdfs_combined.json'
output_dir='/content/drive/MyDrive/chunks'

def chunk_text(text,max_tokens=300,overlap=50):
    sentences = sent_tokenize(text)#tokenising and split in sentences
    chunks=[]
    chunk=[]
    tokens=0

    for sentence in sentences:
        sentence_tokens = len(sentence.split())#counts number of tokens in sentence by splitting on white space
        if tokens+sentence_tokens>max_tokens:#if adding the sentence would be more than 300 finalise the current chunk and save it
            chunks.append(" ".join(chunk))
            if current_chunk:
              num_sentences_to_keep=overlap//len(current_chunk)#adding overlap becaause we are repeating the last part of the chunk in new chunk so model remembers
              current_chunk = current_chunk[-num_sentences_to_keep:] #list slicing from end
            else:
              current_chunk=[]
            tokens = sum(len(s.split()) for s in chunk)#recalculates how many words are there in the current chunk after overlap is complete
        chunk.append(sentence)#adding current sentence to chunk
        tokens=tokens+sentence_tokens
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks
with open(input, "r", encoding="utf-8") as f:
    pdfs_data = json.load(f)

for pdf in pdfs_data:
    filename = pdf["filename"]
    full_text = pdf["text"]
    chunks = chunk_text(full_text)
    output_filename = f"{os.path.splitext(filename)[0]}_chunks.json"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)
