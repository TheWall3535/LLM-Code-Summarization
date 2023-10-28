# Importing libraries
import openai # LLM for code summarization
import tiktoken # Counting tokens
import nltk # calculating BLEU score
import numpy as np # statistical calculations and picking the best summary
import time # waiting between calls to API - an alternative approach would be using tenacity




class dictionary(dict):
    ''' Helper class that makes it easier to add key:value pairs to a dictionary.
    In our case, it is useful for configuring our OpenAI instance'''
    def __init__(self):
        self = dict()
    def add(self, key, value):
        '''add key:value pair to dictionary'''
            self[key] = value


# So far, the main code imports a code file rather than receiving it from a front end.
# With further front end development, this function may not be necessary   
def read_file(filename):
    ''' Helper function to read in a file. For example, can be used to read in the file containing the code needing to be summarized'''
        with open (filename) as file:
            rfile = file.read()
        return rfile


class CodeSummarizer(openai.ChatCompletion):
    ''' Custom subclass of openai.ChatCompletion
    This object provides low-code summarization of Qlikview code via the ```summarize``` method'''

    def __init__(self, auth_params, max_tokens, prompt_length, language = 'qlikview'):
        # openAI authentication parameters
        self.auth_params = auth_params
        self.AZURE_MODEL_NAME = self.auth_params['AZURE_MODEL_NAME']
        self.MODEL = self.auth_params['MODEL']

        # max_tokens refers to the combined token limit between prompt and response(s). For example, a 200 token prompt, and 3 responses of 500 tokens each would = 1700 tokens
        self.max_tokens = max_tokens
        self.language = language
        # This is a dictionary of example inputs (code) and outputs (summaries) that are read from text files
        # A key way to improve summary quality is to improve the quality of these examples
        self.examples = self.__get_examples()

        # The max token limit for each code chunk (if using the preprocessing module to chunk the input code, this should match the ```PROMPT_LENGTH``` variable
        # In this module, it is used as a guide length when the summaries get converted back into code (a step in the summary scoring)
        self.prompt_length = prompt_length
        
        # Stores summaries of code chunks
        self.summaries = None

        # Stores a running summary (higher level summary from lower level summaries)
        self.running_summary = None


    # The principle method of the CodeSummarizer class, takes a ```Chunk``` of code and returns a natural language summary
    # 1) produce candidate summaries 2) convert each summary back into code 3) use a non-contextual text similarity algorithm to evaluate similarity to the original code 4) pick the summary which generated code most resembling the original
    def summarize(self, Chunk, temperature = 1.0, logit_bias = {}, n = 3):
        print('creating summary candidates')

        # makes an LLM API call using the prompt engineering present in the ```__get_candidate_summaries``` method
        summaries = self.__get_candidate_summaries(Chunk = Chunk, temperature = temperature, logit_bias = logit_bias, n = n)
        print('summary candidates created')

        # waits before the next API call - this is a bottleneck caused by our OpenAI instance's token rate limit. May not be necessary for your use case
        time.sleep(70)

        # score each candidate summary by converting the summary back into code (prompt engineering in __get_code_from_summary() method, which is called by the __score_summary() method), and then scoring the resulting code by its similarity to the original code. 

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

    

    # returns a list of generated summaries from a single chunk of code
    def __get_candidate_summaries(self, Chunk, temperature = 1.0, logit_bias = {}, n = 3):

        # calls the create method of openai.ChatCompletion
        # prompt engineering asks for a detailed low level summary of Qlikview code, and includes example input and output (from text files)
        # max tokens for the LLM response is calculated by taking the overall token limit, and subtracting the tokens in the input code chunk, and tokens in the example files
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

    # Given original code and a summary of that code, generate code from the summary,
    # return a list of the scores corresponding to each generated code 
    def __score_summary(self, summary, original_code, generated_code_temperature = 0, generated_codes=1, max_tokens_add=300, metric='bleu'):
        # list of generated codes
        generated_codes = self.__get_code_from_summary(summary, generated_code_temperature, generated_codes, max_tokens_add)

        scores = []
        for gen_code in generated_codes:
            scores.append(self.__get_bleu_score(original_code, gen_code))

        return scores


     # takes a summary, and outputs generated code
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

    # takes original and generated code, then returns the bleu score of the generated code when compared with the original code
    def __get_bleu_score(self, original_code, generated_code):
        reference = nltk.wordpunct_tokenize(original_code.lower())
        candidate = nltk.wordpunct_tokenize(generated_code.lower())
        score = nltk.translate.bleu_score.sentence_bleu([reference], candidate)
        return score
    
    def __get_examples(self):
        if self.language.lower() == 'qlikview':
            low_example_input = open('Qlikview_Example_Low_Level_Input.txt').read()
            low_example_output = open('Qlikview_Example_Low_Level_Output.txt').read()
            hi_example_input = open('Qlikview_Example_High_Level_Input.txt').read()
            hi_example_output = open('Qlikview_Example_High_Level_Output.txt').read()
        return {'low_level_input': low_example_input, 'low_level_output': low_example_output,
               'high_level_input': hi_example_input, 'high_level_output': hi_example_output}

    # helper function which returns the token count for a body of text, as determined by the credentialed LLM
    def count_tokens(self, Text):
        encoding = tiktoken.encoding_for_model(self.MODEL)
        tokens = encoding.encode(Text)
        return len(tokens)

    # stores the summary as the next entry in the instance variable summaries (which is a dictionary)
    def remember_summary(self, summary):
        if self.summaries == None:
            self.summaries = {}
            self.summaries[0] = summary
        else:
            self.summaries[len(self.summaries)] = summary


    # Needs significant prompt engineering and testing, but in theory this uses a summary to update the running summary stored in the running_summary instance variable. This is useful when creating higher level summaries from lower level summaries.
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
       

    