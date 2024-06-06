import json
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import JSONLoader
from pathlib import Path

class Ingestor:
    def __init__(self):
        self.index = None
        self.contexts = []
        # self.questions = []
        # self.answers = []

    def ingest_squad(self, squad_data_file_path='./datasets/dev.json'):
        # load contexts
        contexts_loader = JSONLoader(
            file_path=squad_data_file_path,
            jq_schema='.data[].paragraphs[].context',
            text_content=False)

        contexts = contexts_loader.load()

        self.index = Chroma.from_documents(documents=contexts, embedding=OpenAIEmbeddings())
        
    def get_chroma_index(self):
        return self.index