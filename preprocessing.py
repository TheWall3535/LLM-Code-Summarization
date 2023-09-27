import re
import tiktoken

class FileProcessor():

    def __init__(self, file, PROMPT_LENGTH = 1200, tokenizer_MODEL = 'gpt-3.5-turbo-16k'):
        self.file = file
        self.clean_file = self.read_and_clean()
        self.splits_to_avoid = self.split_index()
        self.tokenizer_MODEL = tokenizer_MODEL
        self.PROMPT_LENGTH = PROMPT_LENGTH
    
    def count_tokens(self, Text):
        encoding = tiktoken.encoding_for_model(self.tokenizer_MODEL)
        tokens = encoding.encode(Text)
        return len(tokens)

    def read_and_clean(self):
    
        with open(self.file) as f:
            lines = f.readlines()
    
        rawCode = [i.lower() for i in lines if (i != "\n" and not i.startswith("//"))]
        result_string = ''.join(rawCode)
        
            #disregard multiline commented sections - unfeasible to not split into chunks that divide commented sections
        for pattern in re.findall(r'/\*(.*?)\*/',result_string,flags = re.DOTALL):
            pattern = r"/*"+pattern+r"*/"
            result_string = re.sub(re.escape(pattern), '', result_string)
        
        return result_string


    def split_index(self):
        # takes single file as string
        
        clean_code_sections = self.clean_file.split("\n")
        
        # if for 
        tuples = [(n,i) for n,i in enumerate(clean_code_sections) if i.strip().startswith("next") or (i.strip().startswith("for") and " = " in i)]
    
        starters = ["for"]
        closers = ["next"]
    
    #     QV_exceptions = []
    
        algo_count = 0
        prev_i = []
        indicies = []
        for i,j in tuples:
            start = re.findall('|'.join(starters),j)
            close = re.findall('|'.join(closers),j)
            if any(start) and any(close):
                continue
            elif any(start):
                if algo_count == 0:
                    key = i
                    # print(i,j)
                if any(prev_i):
                    # print(prev_i,start)
                    algo_count += 1
                else:
                    algo_count+=1
                if i > 0:
                    prev_i = start
                # print(algo_count,i,j)
            elif any(close):
                algo_count -= 1
                if algo_count == 0:
                    # print(i,j)
                    for k in range(key,i):
                        indicies.append(k)
                if i > 0:
                    prev_i = start
                # print(algo_count,i,j)
                
        return indicies

    def make_prompts(self):
        sized_prompts = [""]
        index = 0
        runningChunkTotal = 0
        line_counter = 0
        
        clean_code_sections = self.clean_file.split(";")
        splits_to_avoid = self.splits_to_avoid

        for chunk in clean_code_sections:
            t = self.count_tokens(Text = chunk)
            runningChunkTotal += t

            if runningChunkTotal > self.PROMPT_LENGTH and not line_counter - 1 in splits_to_avoid:
                runningChunkTotal = t 
                index += 1
                sized_prompts.append("")
                # print(runningChunkTotal)
                sized_prompts[index] += (chunk + "\n")
            else:
                # print(runningChunkTotal,"HERE")
                sized_prompts[index] += (chunk + "\n")
                
            line_counter += 1
            
        return sized_prompts


