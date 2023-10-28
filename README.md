# LLM-for-code-translations

## Description: 
* WHAT: Program that produces natural language summaries of Qlikview code.
* WHY: We want a tool that can help Qlikview developers rapidly document code, or help understand legacy code. This would reduce technical debt, and reduce turnaround time on developing new features, fixing bugs, or other end user requests. More broadly, our code could be developed to scale to other languages and use cases
* HOW: We have a short main code which imports from two modules, ```preprocessing```, and ```bradyai```. These modules handle the cleaning and chunking of the original code, and the summarization of said code, respectively.
* WHO: The code was developed by BI Developer Anthony Rautmann (https://github.com/anthonyJamRau) and Marketing Data Science Analyst Wally Castelaz (https://github.com/TheWall3535) of Brady Corporation (bradyid.com)

## Dependencies:
##### Python Libraries
See Dependencies.txt file in repository for installation instructions

##### Files
You will need text files with example input (code) and output (summaries) for both low and high level summaries. 

You will also need text files storing LLM authentication credentials

## Future Development:
Scale code to make use of other LLMs besides OpenAI's GPT-3.5
Scale code and prompt engineering to handle languages besides Qlikview
Develop a front end component - for example, an app where users can submit code into a text box, and have summaries returned to them
