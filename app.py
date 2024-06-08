from ingest import Ingestor
from query import Query

# create index
accessor = Ingestor()
accessor.ingest_squad()
index = accessor.get_chroma_index()
qa_pairs = accessor.get_qa_pairs()

# create query engine
# querier = Query(index)
# query = 'from whence came da vikings that did one time invade paris?'
# result = querier.invoke_query_chain(query)
# print('result', result)