API keys needed
- OPENAI_API_KEY
- LANGSMITH_API_KEY

To run eval
- `pip install -r requirements.txt`
- `python setup_eval.py` (only needs to be run on initial setup. Will persist dataset in Langsmith that can be reused)
- `python eval.py`

To update settings
- Change variables in variables.py. This will update the model used, for example, and also include that metadata in the Langsmith request. This can then be used to identify properties of a given eval run. NOTE: this is not set up for all settings.