from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from langchain_core.runnables import RunnablePassthrough

class Query:
    def __init__(self, index):
        self.llm = ChatOpenAI(temperature=0)
        self.retriever = MultiQueryRetriever.from_llm(
            retriever=index.as_retriever(), llm=self.llm
        )

    # Rank documents by similarity score
    @staticmethod
    def rank_documents(results, k=60):
        fused_scores = {}

        for docs in results:
            for rank, doc in enumerate(docs):
                doc_str = dumps(doc)
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                fused_scores[doc_str] += 1 / (rank + k)

        reranked_results = [
            (loads(doc), score)
            for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]

        return reranked_results


    def get_prompt_perspectives(self, question):
        template = f"""You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from a vector
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search.
        Provide these alternative questions separated by newlines. Original question: {question}"""
        return ChatPromptTemplate.from_template(template)
    
    # Generate multi-query
    def get_query_generator(self, question):
        return (
            self.get_prompt_perspectives(question)
            | ChatOpenAI(temperature=0)
            | StrOutputParser()
            | (lambda x: x.split("\n"))
        )
    
    # Retrieve documents based on generated queries
    def get_retrieval_chain(self, question):
        return self.get_query_generator(question) | self.retriever.map() | self.rank_documents

    def invoke_query_chain(self, question):
        template = """Answer the following question based on this context:

        {context}

        Question: {question}
        """

        prompt = ChatPromptTemplate.from_template(template)

        retrieval_chain = self.get_retrieval_chain(question)

        # Chain
        final_rag_chain = (
            {"context": retrieval_chain, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return final_rag_chain.invoke({"question": question})
