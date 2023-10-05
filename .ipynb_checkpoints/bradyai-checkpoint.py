import openai
import tiktoken
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential
)
import nltk
import numpy as np
import time 

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

class Summary(str):
    def __init__(self, summary, language, prompt):
        super().__init__()
        self.summary = summary
        self.language = language
        self.prompt = prompt


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
        self.summaries = None
        self.running_summary = None

    def __get_examples(self):
        if self.language.lower() == 'qlikview':
            low_example_input = open('Qlikview_Example_Low_Level_Input.txt').read()
            low_example_output = open('Qlikview_Example_Low_Level_Output.txt').read()
            hi_example_input = open('Qlikview_Example_High_Level_Input.txt').read()
            hi_example_output = open('Qlikview_Example_High_Level_Output.txt').read()
        return {'low_level_input': low_example_input, 'low_level_output': low_example_output,
               'high_level_input': hi_example_input, 'high_level_output': hi_example_output}

    def count_tokens(self, Text):
        encoding = tiktoken.encoding_for_model(self.MODEL)
        tokens = encoding.encode(Text)
        return len(tokens)
        
    @retry(wait=wait_random_exponential(min=1,max=60),stop=stop_after_attempt(6))
    def __completion_with_backoff(**kwargs):
        return self.create(**kwargs)


    # returns a list of generated summaries from a single chunk of code
    def __get_candidate_summaries(self, Chunk, temperature = 1.0, logit_bias = {}, n = 3):
        summaries = self.create(engine = self.AZURE_MODEL_NAME,
                                              messages = [{"role": "system", "content" : f"You are giving step by step low level detail of {self.language} codes so that developers will not have to reference the source code after reading your response."},
            {"role": "user", "content": f"Be extremely specific about any conditions that are in the code. \n code:\n```\n {self.examples['low_level_input']} \n```\n"},
            {"role": "assistant", "content" : f"{self.examples['low_level_output']}"},
            {"role": "user", "content": f"Be extremely specific about any conditions that are in the code.  \n code:\n```\n {Chunk} \n```\n"}
                ],
                                              max_tokens = (self.max_tokens - (self.count_tokens(Chunk) + self.count_tokens(self.examples['low_level_input']) + self.count_tokens(self.examples['low_level_output']))),
        temperature = temperature,
        logit_bias = logit_bias,
        n = n)

        return [choice['message']['content'] for choice in summaries['choices']]

    # takes a summary, and outputs generated code for each summary
    def __get_code_from_summary(self, summary, temperature = 0, n=1, max_tokens_add = 300):
        generated_code = self.create(
                engine = self.AZURE_MODEL_NAME,
                messages = [
                        {"role": "user", "content": f"Use the following summary of {self.language} code to generate a script of {self.language} code WITHOUT COMMENTS IN IT. Return only the code. \n summary:\n```\n {summary} \n```\n"},
                            ],
                max_tokens = self.prompt_length+max_tokens_add,
                temperature = temperature,
                n = n
            )

        return [choice['message']['content'] for choice in generated_code['choices']]

    def __get_bleu_score(self, original_code, generated_code):
        reference = nltk.wordpunct_tokenize(original_code.lower())
        candidate = nltk.wordpunct_tokenize(generated_code.lower())
        score = nltk.translate.bleu_score.sentence_bleu([reference], candidate)
        return score

    # Given original code and a summary of that code, generate code from the summary,
    # return a list of the scores corresponding to each generated code 
    def __score_summary(self, summary, original_code, generated_code_temperature = 0, generated_codes=1, max_tokens_add=300, metric='bleu'):
        # list of generated codes
        generated_codes = self.__get_code_from_summary(summary, generated_code_temperature, generated_codes, max_tokens_add)

        scores = []
        for gen_code in generated_codes:
            scores.append(self.__get_bleu_score(original_code, gen_code))

        return scores

  
    def summarize(self, Chunk, temperature = 1.0, logit_bias = {}, n = 3):
        print('creating summary candidates')
        summaries = self.__get_candidate_summaries(Chunk = Chunk, temperature = temperature, logit_bias = logit_bias, n = n)
        print('summary candidates created')

        time.sleep(70)

        summary_scoring = {}
        for summary_idx in range(0, len(summaries)):
            print(f'scoring summary {summary_idx + 1} of {len(summaries)}')
            score_array = np.array(self.__score_summary(summaries[summary_idx], Chunk))
            summary_scoring[summary_idx] = {'summary': summaries[summary_idx],
                                        'mean': score_array.mean(),
                                        'max': score_array.max()}
            time.sleep(70)



        best_summary_idx = np.argmax([summary_scoring[x]['mean'] for x in summary_scoring])
        best_summary = summary_scoring[best_summary_idx]
        best_summary['original_code'] = Chunk

        
        return best_summary

    def remember_summary(self, summary):
        if self.summaries == None:
            self.summaries = {}
            self.summaries[0] = summary
        else:
            self.summaries[len(self.summaries)] = summary


    def update_running_summary(self, summary, temperature = 0.2, logit_bias = {}, n = 1):
        print('updating summary')
        if self.running_summary == None:
            self.running_summary = summary
        else:
            updated_summary = self.create(engine = self.AZURE_MODEL_NAME,
                                              messages = [{"role": "system", "content" : f"You are creating a high level summary, from low level summaries of {self.language} codes."},
            {"role": "user", "content": f"Update the existing high level summary, with the following low level summary  {self.examples['high_level_input']}"},
            {"role": "assistant", "content" : f"{self.examples['high_level_output']}"},
            {"role": "user", "content": f"Update the existing high level summary, with the following low level summary \n High Level Summary:\n```\n {self.running_summary} \n```\n Low Level Summary:\n```\n {summary} \n```\n"}
                ],
                                              max_tokens = (self.max_tokens - (self.count_tokens(self.running_summary))),
        temperature = temperature,
        logit_bias = logit_bias,
        n = n)
            self.running_summary = updated_summary

        return self.running_summary


    def clear_running_summary(self):
        self.running_summary = None
        print('running summary cleared')
       

    