from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import JSONLoader
import json

class Ingestor:
    def __init__(self):
        self.index = None
        self.qa_pairs = []

    def ingest_squad(self, squad_data_file_path='./datasets/dev.json'):
        # Load contexts using JSONLoader
        contexts_loader = JSONLoader(
            file_path=squad_data_file_path,
            jq_schema='.data[].paragraphs[].context',
            text_content=False)
        
        contexts = contexts_loader.load()

        # Load the entire JSON data to extract QA pairs
        with open(squad_data_file_path, 'r') as file:
            data = json.load(file)
        
        qa_pairs = []
        
        for item in data["data"]:
            for paragraph in item["paragraphs"]:
                for qa in paragraph["qas"]:
                    question = qa["question"]
                    answers = qa.get("answers", [])
                    qa_pairs.append({
                        "question": question,
                        "answers": answers
                    })
        
        # Create the vector index for contexts
        self.index = Chroma.from_documents(documents=contexts, embedding=OpenAIEmbeddings())
        self.qa_pairs = qa_pairs
        
    def get_chroma_index(self):
        return self.index

    def get_qa_pairs(self):
        return self.qa_pairs
