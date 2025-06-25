import fitz
import os
import json
from google.colab import drive
drive.mount('/content/drive')
pdf_dir = "/content/drive/MyDrive/QuantGPT"
all_pdfs_data = []
for filename in os.listdir(pdf_dir):#checking every pdf in the folder
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_dir, filename)
        doc = fitz.open(pdf_path)#opening every pdf
        all_text = ""
        for page_num, page in enumerate(doc):#getting text pagenumber wise to store in json
            all_text =all_text+f"\n\n--- Page {page_num + 1} ---\n{page.get_text()}"
        pdf_data = {
            "filename": filename,
            "text": all_text,
            "pages": len(doc)
        }
        all_pdfs_data.append(pdf_data)#appending data
#saved all in one file 
output_json = "all_pdfs_combined.json"
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(all_pdfs_data, f, indent=2)
