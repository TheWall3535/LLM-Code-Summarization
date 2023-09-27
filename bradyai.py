import openai
import tiktoken
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential
)

class dictionary(dict):
    def __init__(self):
        self = dict()
    def add(self, key, value):
            self[key] = value


def read_file(filename):
        with open (filename) as file:
            rfile = file.read()
        return rfile

params = dictionary()
for line in read_file("GPT_AUTH (1).txt").split("\n"):
    sep = line.split(" ")
    params.add(sep[0],sep[1])

openai.api_base = params['OPENAI_API_BASE']
openai.api_key = params['OPENAI_API_KEY']
openai.api_version = params["OPENAI_API_VERSION"]
openai.api_type = params["API_TYPE"]


class CodeSummarizer(openai.ChatCompletion):

    def __init__(self, auth_params, max_tokens, prompt_length, language = 'qlikview'):
        self.auth_params = auth_params
        self.AZURE_MODEL_NAME = self.auth_params['AZURE_MODEL_NAME']
        self.MODEL = self.auth_params['MODEL']
        self.max_tokens = max_tokens
        self.language = language
        self.examples = self.__get_examples()
        self.prompt_length = prompt_length
        # DRAFTING
        self.running_summary = ''

    def __get_examples(self):
        if self.language.lower() == 'qlikview':
            example_input = open('Qlikview_Example_Input.txt').read()
            example_output = open('Qlikview_Example_Output.txt').read()
        return {'input': example_input, 'output': example_output}

    def count_tokens(self, Text):
        encoding = tiktoken.encoding_for_model(self.MODEL)
        tokens = encoding.encode(Text)
        return len(tokens)
        
    @retry(wait=wait_random_exponential(min=1,max=60),stop=stop_after_attempt(6))
    def __completion_with_backoff(**kwargs):
        return self.create(**kwargs)

    def summarize(self, Chunk, temperature = 1.0, logit_bias = {}, n = 3):
        self.create(engine = self.AZURE_MODEL_NAME,
                                              messages = [{"role": "system", "content" : "You are giving step by step low level detail of Qlikview codes so that developers will not have to reference the source code after reading your response."},
            {"role": "user", "content": f"Be extremely specific about any conditions that are in the code. \n code:\n```\n {self.examples['input']} \n```\n"},
            {"role": "assistant", "content" : f"{self.examples['output']}"},
            {"role": "user", "content": f"Be extremely specific about any conditions that are in the code.  \n code:\n```\n {Chunk} \n```\n"}
                ],
                                              max_tokens = (self.max_tokens - (self.count_tokens(Chunk) + self.count_tokens(self.examples['input']) + self.count_tokens(self.examples['output']))),
        temperature = temperature,
        logit_bias = logit_bias,
        n = 3)

        # pick the best summary
        Bleu...


        # update the running summary
        call to another method
        self.running_summary += winning_summary

        return winning_summary, 

    def update_running_summary(self):


        
