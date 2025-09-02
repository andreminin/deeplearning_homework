# Tagging and Extraction Using OpenAI functions


```python
import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']
```


```python
from typing import List
from pydantic import BaseModel, Field
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
```


```python
class Tagging(BaseModel):
    """Tag the piece of text with particular info."""
    sentiment: str = Field(description="sentiment of text, should be `pos`, `neg`, or `neutral`")
    language: str = Field(description="language of text (should be ISO 639-1 code)")
```


```python
convert_pydantic_to_openai_function(Tagging)
```




    {'name': 'Tagging',
     'description': 'Tag the piece of text with particular info.',
     'parameters': {'title': 'Tagging',
      'description': 'Tag the piece of text with particular info.',
      'type': 'object',
      'properties': {'sentiment': {'title': 'Sentiment',
        'description': 'sentiment of text, should be `pos`, `neg`, or `neutral`',
        'type': 'string'},
       'language': {'title': 'Language',
        'description': 'language of text (should be ISO 639-1 code)',
        'type': 'string'}},
      'required': ['sentiment', 'language']}}




```python
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
```


```python
model = ChatOpenAI(temperature=0)
```


```python
tagging_functions = [convert_pydantic_to_openai_function(Tagging)]
```


```python
tagging_functions
```




    [{'name': 'Tagging',
      'description': 'Tag the piece of text with particular info.',
      'parameters': {'title': 'Tagging',
       'description': 'Tag the piece of text with particular info.',
       'type': 'object',
       'properties': {'sentiment': {'title': 'Sentiment',
         'description': 'sentiment of text, should be `pos`, `neg`, or `neutral`',
         'type': 'string'},
        'language': {'title': 'Language',
         'description': 'language of text (should be ISO 639-1 code)',
         'type': 'string'}},
       'required': ['sentiment', 'language']}}]




```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "Think carefully, and then tag the text as instructed"),
    ("user", "{input}")
])
```


```python
model_with_functions = model.bind(
    functions=tagging_functions,
    function_call={"name": "Tagging"}
)
```


```python
tagging_chain = prompt | model_with_functions
```


```python
tagging_chain.invoke({"input": "I love langchain"})
```




    AIMessage(content='', additional_kwargs={'function_call': {'name': 'Tagging', 'arguments': '{"sentiment":"pos","language":"en"}'}})




```python
tagging_chain.invoke({"input": "non mi piace questo cibo"})
```




    AIMessage(content='', additional_kwargs={'function_call': {'name': 'Tagging', 'arguments': '{"sentiment":"neg","language":"it"}'}})




```python
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
```


```python
tagging_chain = prompt | model_with_functions | JsonOutputFunctionsParser()
```


```python
tagging_chain.invoke({"input": "non mi piace questo cibo"})
```




    {'sentiment': 'neg', 'language': 'it'}



## Extraction

Extraction is similar to tagging, but used for extracting multiple pieces of information.


```python
from typing import Optional
class Person(BaseModel):
    """Information about a person."""
    name: str = Field(description="person's name")
    age: Optional[int] = Field(description="person's age")
```


```python
class Information(BaseModel):
    """Information to extract."""
    people: List[Person] = Field(description="List of info about people")
```


```python
convert_pydantic_to_openai_function(Information)
```




    {'name': 'Information',
     'description': 'Information to extract.',
     'parameters': {'title': 'Information',
      'description': 'Information to extract.',
      'type': 'object',
      'properties': {'people': {'title': 'People',
        'description': 'List of info about people',
        'type': 'array',
        'items': {'title': 'Person',
         'description': 'Information about a person.',
         'type': 'object',
         'properties': {'name': {'title': 'Name',
           'description': "person's name",
           'type': 'string'},
          'age': {'title': 'Age',
           'description': "person's age",
           'type': 'integer'}},
         'required': ['name']}}},
      'required': ['people']}}




```python
extraction_functions = [convert_pydantic_to_openai_function(Information)]
extraction_model = model.bind(functions=extraction_functions, function_call={"name": "Information"})
```


```python
extraction_model.invoke("Joe is 30, his mom is Martha")
```




    AIMessage(content='', additional_kwargs={'function_call': {'name': 'Information', 'arguments': '{"people":[{"name":"Joe","age":30},{"name":"Martha"}]}'}})




```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract the relevant information, if not explicitly provided do not guess. Extract partial info"),
    ("human", "{input}")
])
```


```python
extraction_chain = prompt | extraction_model
```


```python
extraction_chain.invoke({"input": "Joe is 30, his mom is Martha"})
```




    AIMessage(content='', additional_kwargs={'function_call': {'name': 'Information', 'arguments': '{"people":[{"name":"Joe","age":30},{"name":"Martha"}]}'}})




```python
extraction_chain = prompt | extraction_model | JsonOutputFunctionsParser()
```


```python
extraction_chain.invoke({"input": "Joe is 30, his mom is Martha"})
```




    {'people': [{'name': 'Joe', 'age': 30}, {'name': 'Martha'}]}




```python
from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser
```


```python
extraction_chain = prompt | extraction_model | JsonKeyOutputFunctionsParser(key_name="people")
```


```python
extraction_chain.invoke({"input": "Joe is 30, his mom is Martha"})
```




    [{'name': 'Joe', 'age': 30}, {'name': 'Martha'}]



## Doing it for real

We can apply tagging to a larger body of text.

For example, let's load this blog post and extract tag information from a sub-set of the text.


```python
from langchain.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
documents = loader.load()
```


```python
doc = documents[0]
```


```python
page_content = doc.page_content[:10000]
```


```python
print(page_content[:1000])
```

    
    
    
    
    
    
    LLM Powered Autonomous Agents | Lil'Log
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    Lil'Log
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    |
    
    
    
    
    
    
    Posts
    
    
    
    
    Archive
    
    
    
    
    Search
    
    
    
    
    Tags
    
    
    
    
    FAQ
    
    
    
    
    
    
    
    
    
          LLM Powered Autonomous Agents
        
    Date: June 23, 2023  |  Estimated Reading Time: 31 min  |  Author: Lilian Weng
    
    
     
    
    
    Table of Contents
    
    
    
    Agent System Overview
    
    Component One: Planning
    
    Task Decomposition
    
    Self-Reflection
    
    
    Component Two: Memory
    
    Types of Memory
    
    Maximum Inner Product Search (MIPS)
    
    
    Component Three: Tool Use
    
    Case Studies
    
    Scientific Discovery Agent
    
    Generative Agents Simulation
    
    Proof-of-Concept Examples
    
    
    Challenges
    
    Citation
    
    References
    
    
    
    
    
    Building agents with LLM (large language model) as its core controller is a cool concept. Several proof-of-concepts demos, such as AutoGPT, GPT-Engineer and BabyAGI, serve as inspiring examples. The potentiality of LLM extends beyond generating well-written copies, stories, essays and programs; it can be framed as a powerful general problem solver.
    



```python
class Overview(BaseModel):
    """Overview of a section of text."""
    summary: str = Field(description="Provide a concise summary of the content.")
    language: str = Field(description="Provide the language that the content is written in.")
    keywords: str = Field(description="Provide keywords related to the content.")
```


```python
overview_tagging_function = [
    convert_pydantic_to_openai_function(Overview)
]
tagging_model = model.bind(
    functions=overview_tagging_function,
    function_call={"name":"Overview"}
)
tagging_chain = prompt | tagging_model | JsonOutputFunctionsParser()
```


```python
tagging_chain.invoke({"input": page_content})
```




    {'summary': 'The article discusses building autonomous agents powered by LLM (large language model) as the core controller. It covers components such as planning, memory, and tool use, along with examples and challenges in implementing LLM-powered agents.',
     'language': 'English',
     'keywords': 'LLM, autonomous agents, planning, memory, tool use, proof-of-concepts, challenges'}




```python
class Paper(BaseModel):
    """Information about papers mentioned."""
    title: str
    author: Optional[str]


class Info(BaseModel):
    """Information to extract"""
    papers: List[Paper]
```


```python
paper_extraction_function = [
    convert_pydantic_to_openai_function(Info)
]
extraction_model = model.bind(
    functions=paper_extraction_function, 
    function_call={"name":"Info"}
)
extraction_chain = prompt | extraction_model | JsonKeyOutputFunctionsParser(key_name="papers")
```


```python
extraction_chain.invoke({"input": page_content})
```




    [{'title': 'LLM Powered Autonomous Agents', 'author': 'Lilian Weng'}]




```python
template = """A article will be passed to you. Extract from it all papers that are mentioned by this article follow by its author. 

Do not extract the name of the article itself. If no papers are mentioned that's fine - you don't need to extract any! Just return an empty list.

Do not make up or guess ANY extra information. Only extract what exactly is in the text."""

prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", "{input}")
])
```


```python
extraction_chain = prompt | extraction_model | JsonKeyOutputFunctionsParser(key_name="papers")
```


```python
extraction_chain.invoke({"input": page_content})
```




    [{'title': 'Chain of thought (CoT; Wei et al. 2022)'},
     {'title': 'Tree of Thoughts (Yao et al. 2023)'},
     {'title': 'LLM+P (Liu et al. 2023)'},
     {'title': 'ReAct (Yao et al. 2023)'},
     {'title': 'Reflexion (Shinn & Labash 2023)'},
     {'title': 'Chain of Hindsight (CoH; Liu et al. 2023)'},
     {'title': 'Algorithm Distillation (AD; Laskin et al. 2023)'}]




```python
extraction_chain.invoke({"input": "hi"})
```




    []




```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_overlap=0)
```


```python
splits = text_splitter.split_text(doc.page_content)
```


```python
len(splits)
```




    15




```python
def flatten(matrix):
    flat_list = []
    for row in matrix:
        flat_list += row
    return flat_list
```


```python
flatten([[1, 2], [3, 4]])
```




    [1, 2, 3, 4]




```python
print(splits[0])
```

    LLM Powered Autonomous Agents | Lil'Log
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    Lil'Log
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    |
    
    
    
    
    
    
    Posts
    
    
    
    
    Archive
    
    
    
    
    Search
    
    
    
    
    Tags
    
    
    
    
    FAQ
    
    
    
    
    
    
    
    
    
          LLM Powered Autonomous Agents
        
    Date: June 23, 2023  |  Estimated Reading Time: 31 min  |  Author: Lilian Weng
    
    
     
    
    
    Table of Contents
    
    
    
    Agent System Overview
    
    Component One: Planning
    
    Task Decomposition
    
    Self-Reflection
    
    
    Component Two: Memory
    
    Types of Memory
    
    Maximum Inner Product Search (MIPS)
    
    
    Component Three: Tool Use
    
    Case Studies
    
    Scientific Discovery Agent
    
    Generative Agents Simulation
    
    Proof-of-Concept Examples
    
    
    Challenges
    
    Citation
    
    References
    
    
    
    
    
    Building agents with LLM (large language model) as its core controller is a cool concept. Several proof-of-concepts demos, such as AutoGPT, GPT-Engineer and BabyAGI, serve as inspiring examples. The potentiality of LLM extends beyond generating well-written copies, stories, essays and programs; it can be framed as a powerful general problem solver.
    Agent System Overview#
    In a LLM-powered autonomous agent system, LLM functions as the agent’s brain, complemented by several key components:
    
    Planning
    
    Subgoal and decomposition: The agent breaks down large tasks into smaller, manageable subgoals, enabling efficient handling of complex tasks.
    Reflection and refinement: The agent can do self-criticism and self-reflection over past actions, learn from mistakes and refine them for future steps, thereby improving the quality of final results.
    
    
    Memory
    
    Short-term memory: I would consider all the in-context learning (See Prompt Engineering) as utilizing short-term memory of the model to learn.
    Long-term memory: This provides the agent with the capability to retain and recall (infinite) information over extended periods, often by leveraging an external vector store and fast retrieval.
    
    
    Tool use
    
    The agent learns to call external APIs for extra information that is missing from the model weights (often hard to change after pre-training), including current information, code execution capability, access to proprietary information sources and more.
    
    
    
    
    
    Overview of a LLM-powered autonomous agent system.



```python
from langchain.schema.runnable import RunnableLambda
```


```python
prep = RunnableLambda(
    lambda x: [{"input": doc} for doc in text_splitter.split_text(x)]
)
```


```python
prep.invoke("hi")
```




    [{'input': 'hi'}]




```python
chain = prep | extraction_chain.map() | flatten
```


```python
chain.invoke(doc.page_content)
```




    [{'title': 'AutoGPT'},
     {'title': 'GPT-Engineer'},
     {'title': 'BabyAGI'},
     {'title': 'Chain of thought', 'author': 'Wei et al. 2022'},
     {'title': 'Tree of Thoughts', 'author': 'Yao et al. 2023'},
     {'title': 'LLM+P', 'author': 'Liu et al. 2023'},
     {'title': 'ReAct', 'author': 'Yao et al. 2023'},
     {'title': 'Reflexion', 'author': 'Shinn & Labash 2023'},
     {'title': 'Chain of Hindsight (CoH; Liu et al. 2023'},
     {'title': 'Algorithm Distillation (AD; Laskin et al. 2023'},
     {'title': 'Duan et al. 2017'},
     {'title': 'Laskin et al. 2023'},
     {'title': 'Miller 1956'},
     {'title': 'LSH: Locality-Sensitive Hashing', 'author': 'Andoni, Alexandr'},
     {'title': 'MRKL (Karpas et al. 2022)', 'author': 'Karpas et al.'},
     {'title': 'TALM (Tool Augmented Language Models; Parisi et al. 2022)',
      'author': 'Parisi et al.'},
     {'title': 'Toolformer (Schick et al. 2023)', 'author': 'Schick et al.'},
     {'title': 'HuggingGPT (Shen et al. 2023)', 'author': 'Shen et al.'},
     {'title': 'API-Bank', 'author': 'Li et al. 2023'},
     {'title': 'ChemCrow', 'author': 'Bran et al. 2023'},
     {'title': 'Boiko et al. (2023)'},
     {'title': 'Generative Agents Simulation', 'author': 'Park, et al. 2023'},
     {'title': 'AutoGPT: A Proof-of-Concept for Autonomous Agents with Large Language Models as Controllers'},
     {'title': 'GPT-Engineer', 'author': 'OpenAI'},
     {'title': 'A Survey of Machine Learning Algorithms', 'author': 'John Smith'},
     {'title': 'Deep Learning Techniques for Image Recognition',
      'author': 'Emily Brown'},
     {'title': 'Python toolbelt preferences'},
     {'title': 'Chain of thought prompting elicits reasoning in large language models'},
     {'title': 'Tree of Thoughts: Deliberate Problem Solving with Large Language Models'},
     {'title': 'Chain of Hindsight Aligns Language Models with Feedback'},
     {'title': 'LLM+P: Empowering Large Language Models with Optimal Planning Proficiency'},
     {'title': 'ReAct: Synergizing reasoning and acting in language models'},
     {'title': 'Reflexion: an autonomous agent with dynamic memory and self-reflection'},
     {'title': 'In-context Reinforcement Learning with Algorithm Distillation'},
     {'title': 'MRKL Systems A modular, neuro-symbolic architecture that combines large language models, external knowledge sources and discrete reasoning'},
     {'title': 'Webgpt: Browser-assisted question-answering with human feedback'},
     {'title': 'Tool Augmented Language Models'},
     {'title': 'Toolformer: Language Models Can Teach Themselves to Use Tools'},
     {'title': 'API-Bank: A Benchmark for Tool-Augmented LLMs'},
     {'title': 'HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in HuggingFace'},
     {'title': 'ChemCrow: Augmenting large-language models with chemistry tools'},
     {'title': 'Emergent autonomous scientific research capabilities of large language models'},
     {'title': 'Generative Agents: Interactive Simulacra of Human Behavior'}]




```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
