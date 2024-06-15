This is a tool that uses the [SQuAD Dataset](https://rajpurkar.github.io/SQuAD-explorer/) to evaluate the effectiveness of a RAG pipeline in the following steps:
- adds all of the articles in the SQuAD dataset to a Chroma vector database
- queries the set up RAG pipeline using the questions in the dataset
- compares the answers returned from the RAG pipeline to the correct answers provided in the SQuAD dataset

There are two separate query files, `query_basic`, which uses a very straightforward RAG pipeline, and `query`, which involves task decomposition and reranking. 
For initial testing, I would recommend using `query_basic`, as there are over 11k questions in the dataset, and about 5 LLM calls per question, so the whole process can get lengthy and expensive quite quickly.

API keys needed
- `OPENAI_API_KEY`
- `LANGSMITH_API_KEY`

To run eval
- `pip install -r requirements.txt`
- `python setup_eval.py` (only needs to be run on initial setup. Will persist dataset in Langsmith that can be reused)
- `python eval.py`

To update settings
- Change variables in variables.py. This will update the model used, for example, and also include that metadata in the Langsmith request. This can then be used to identify properties of a given eval run. NOTE: this is not set up for all settings.
