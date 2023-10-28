import openai
from bradyai import CodeSummarizer, dictionary, read_file
from preprocessing import FileProcessor

# configuring OpenAI LLM
params = dictionary()
for line in read_file("credentials.txt").split("\n"):
    sep = line.split(" ")
    params.add(sep[0],sep[1])

openai.api_base = params['OPENAI_API_BASE']
openai.api_key = params['OPENAI_API_KEY']
openai.api_version = params["OPENAI_API_VERSION"]
openai.api_type = params["API_TYPE"]

# Get code input
code_path = "code_file.txt"

# Process raw code

processor = FileProcessor(code_path)
prompts = processor.make_prompts()

# Summarize code
summarizer = CodeSummarizer(params, max_tokens=16000, chunk_length=1200, language='qlikview')
response = summarizer.summarize(prompts[0])

# Save summary to object memory
summarizer.remember_summary(response)
