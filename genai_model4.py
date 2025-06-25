#===================
#All the imports
#===================
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
import re

#===================
# taking user input
#===================

def take_user_input():
    print(" Ask your question?, e.g., Waht is abdonminal surgery?")
    question = input("you: ")
    return question

#===================
# RAG implementation
#===================

# Extracting data out of the pdf
loader = PyMuPDFLoader("Instructions.pdf")
document = loader.load()

# Chunking the data for better retrieval
splitter = RecursiveCharacterTextSplitter(chunk_size = 500, 
                                          chunk_overlap = 50,
                                           separators=["\n\n", "\n", ".", "!", "?", " ", ""])
docs = splitter.split_documents(document)

# Embeding the chunks
embeddings = SentenceTransformerEmbeddings(model_name = "all-MiniLM-L6-v2")

# Storing the vectore in the vector database ( Chroma )
db = Chroma.from_documents(docs, embeddings, persist_directory="./surgiguides_db")

# retrieval of the vectors with meaning from the db
retriever = db.as_retriever()


#===========================================
# The model inference or pipeline formation
#===========================================

model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# pipeline for the llm
flan_pipeline = pipeline("text2text-generation", model = model, tokenizer = tokenizer)
llm = HuggingFacePipeline(pipeline = flan_pipeline)

#=====================
# RAG Pipeline 
#=====================
qa_chain = RetrievalQA.from_chain_type(
    llm = llm, 
    retriever = retriever,
    return_source_documents = True
)

#==============================
# Preprocessing the input data
#==============================
def preprocessing(text):
    text = text.lower()
    text = re.sub(r'/s+',' ',text).strip()
    return text


#=================
# Postprocessing
#=================
def postprocess_output(text):
# Remove redundant whitespace and line breaks
    text = re.sub(r'\s+', ' ', text).strip()
# Capitalize first letter if missing
    if text and not text[0].isupper():
        text = text[0].upper() + text[1:]
# Add a period at the end if missing
    if text and text[-1] not in '.!?':
        text += '.'
# Convert lists into bullet points
    if "step" in text.lower() or "procedure" in text.lower():
        lines = re.split(r'\d+\.', text)
        lines = [line.strip() for line in lines if line.strip()]
        text = "\n• " + "\n• ".join(lines)
    return text

#=========================
# Getting the output
#=========================

if __name__ == "__main__":
    while True:
        query = take_user_input()
        if query.lower() in ["quit","exit"]:
            print("GoodBye")
            break
        cleaned = preprocessing(query)
        print("\n Generating Your answer")
        output = qa_chain.invoke(cleaned)
        postprocessed = postprocess_output(output['result'])
        print("\n📘 Surgiguide says:\n", postprocessed)

