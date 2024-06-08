from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from variables import settings

class Query:
    def __init__(self, index):
        self.index = index
        self.llm = ChatOpenAI(temperature=0, base_url="http://localhost:1234/v1", api_key="lm-studio", model=settings["model"])
        self.retriever = index.as_retriever()

    # Retrieve documents based on the question
    def get_documents(self, question):
        print('in get docs', question)
        return self.retriever.invoke(question)
    
    def invoke_query_chain(self, question):
        print('question in invoke', question)
        template = """Answer the following question based on this context:

        {context}

        Question: {question}

        If no relevant information is found in the question, return the text "No reference found"
        """

        prompt = ChatPromptTemplate.from_template(template)

        # Retrieve documents based on the question
        documents = self.get_documents(question)
        context = "\n".join([doc.page_content for doc in documents])


        # Generate the prompt
        prompt_output = prompt.invoke({"context": context, "question": question})
        # Pass the prompt output to the LLM
        llm_output = self.llm.invoke(prompt_output)
        # Parse the LLM output
        parsed_output = StrOutputParser().invoke(llm_output)

        return parsed_output

# Example usage
# Assuming `index` is already defined and properly set up
# query_instance = Query(index)
# response = query_instance.invoke_query_chain("In what country is Normandy located?")
# print(response)
