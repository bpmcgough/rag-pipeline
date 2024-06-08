# create dataset
from langsmith import Client
from ingest import Ingestor

# TODO: should not need to do this in many different places
accessor = Ingestor()
accessor.ingest_squad()
index = accessor.get_chroma_index()
qa_pairs = accessor.get_qa_pairs()

dataset_inputs = []
dataset_outputs = []

for qa in qa_pairs:
    question = qa['question']
    answers = qa['answers']
    
    dataset_inputs.append(question)
    
    unique_answers = list(set(answer['text'] for answer in answers))
    dataset_outputs.append({'must_mention': unique_answers})

client = Client()
dataset_name = "SQuAD subset"

subset_size = 10
subset_inputs = dataset_inputs[:subset_size]
subset_outputs = dataset_outputs[:subset_size]

dataset = client.create_dataset(
    dataset_name=dataset_name,
    description="Subset of SQuAD",
)
client.create_examples(
    inputs=[{"question": q} for q in subset_inputs],
    outputs=subset_outputs,
    dataset_id=dataset.id,
)