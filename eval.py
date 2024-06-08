from langsmith.schemas import Run, Example
from langsmith.evaluation import evaluate
from query_basic import Query
from ingest import Ingestor
from variables import settings

# TODO: should not need to do this in many different places
# set up querier
accessor = Ingestor()
accessor.ingest_squad()
index = accessor.get_chroma_index()
querier = Query(index)

def must_mention(run: Run, example: Example) -> dict:
    prediction = run.outputs.get("output") or ""
    required = example.outputs.get("must_mention") or []
    if not required:
        return {"key": "must_mention", "score": "No reference found"}
    score = any(phrase in prediction for phrase in required)
    return {"key": "must_mention", "score": score}

evaluators = [
  must_mention,
#   LangChainStringEvaluator(
#     "criteria",
#     config={"criteria":  "harmfulness"}
#   ),
]

def query_wrapper(query_dict):
    query_string = query_dict['question']
    response = querier.invoke_query_chain(query_string)
    return {"output": response}

experiment_results = evaluate(
    query_wrapper,
    data="SQuAD subset",
    evaluators=evaluators,
    experiment_prefix="squad",
    metadata=settings
    # TODO: have metadata stored in one central place that also controls what pieces are used
    #   "model": "gpt-3.5-turbo",
    #   "parser": "pdf-miner",
    #   "reranking model": "by hand",
    #   "chunking strategy": "RecursiveCharacterTextSplitter"
    ,
)