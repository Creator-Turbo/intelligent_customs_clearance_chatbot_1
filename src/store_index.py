import os 
from pinecone.grpc import PineconeGRPC as Pinecone 
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from dotenv import load_dotenv

from src.helper import load_multiple_pdfs ,text_split,download_hugging_face_embeddings
load_dotenv()

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

# Example usage:
pdf_files = [
    "D:\Customs Clearance_chatbot\\data\\Customs Tariff 2024-25_zz1tedk.pdf",
    "D:\Customs Clearance_chatbot\\data\\Navigating Import.pdf",
    "D:\Customs Clearance_chatbot\\data\\nepal.pdf",
    "D:\Customs Clearance_chatbot\\data\\np_e.pdf",
    "D:\Customs Clearance_chatbot\\data\\trade_industry_tax.pdf"
]


extracted_data = load_multiple_pdfs(pdf_files)
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "customs-clearance-chatbot"

pc.create_index(
    name = index_name,
    dimension=384,
    metric="cosine",
    spec = ServerlessSpec(
        cloud = "aws",
        region="us-east-1"
    )
)



# Embed each chunk and upsert the embeddings into your Pinecone index 
docsearch = PineconeVectorStore.from_documents(
    documents = text_chunks,
    index_name = index_name,
    embedding = embeddings,
)
