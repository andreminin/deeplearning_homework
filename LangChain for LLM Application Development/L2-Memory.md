# LangChain: Memory

## Outline
* ConversationBufferMemory
* ConversationBufferWindowMemory
* ConversationTokenBufferMemory
* ConversationSummaryMemory

## ConversationBufferMemory


```python
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

import warnings
warnings.filterwarnings('ignore')
```

Note: LLM's do not always produce the same results. When executing the code in your notebook, you may get slightly different answers that those in the video.


```python
# account for deprecation of LLM model
import datetime
# Get the current date
current_date = datetime.datetime.now().date()

# Define the date after which the model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2024, 6, 12)

# Set the model variable based on the current date
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"
```


```python
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

```


```python
llm = ChatOpenAI(temperature=0.0, model=llm_model)
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=True
)
```


```python
conversation.predict(input="Hi, my name is Andrew")
```

    
    
    [1m> Entering new ConversationChain chain...[0m
    Prompt after formatting:
    [32;1m[1;3mThe following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
    
    Current conversation:
    
    Human: Hi, my name is Andrew
    AI:[0m
    
    [1m> Finished chain.[0m





    "Hello Andrew! It's nice to meet you. How can I assist you today?"




```python
conversation.predict(input="What is 1+1?")
```

    
    
    [1m> Entering new ConversationChain chain...[0m
    Prompt after formatting:
    [32;1m[1;3mThe following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
    
    Current conversation:
    Human: Hi, my name is Andrew
    AI: Hello Andrew! It's nice to meet you. How can I assist you today?
    Human: What is 1+1?
    AI:[0m
    
    [1m> Finished chain.[0m





    '1+1 equals 2. Is there anything else you would like to know?'




```python
conversation.predict(input="What is my name?")
```

    
    
    [1m> Entering new ConversationChain chain...[0m
    Prompt after formatting:
    [32;1m[1;3mThe following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
    
    Current conversation:
    Human: Hi, my name is Andrew
    AI: Hello Andrew! It's nice to meet you. How can I assist you today?
    Human: What is 1+1?
    AI: 1+1 equals 2. Is there anything else you would like to know?
    Human: What is my name?
    AI:[0m
    
    [1m> Finished chain.[0m





    'Your name is Andrew.'




```python
print(memory.buffer)
```

    Human: Hi, my name is Andrew
    AI: Hello Andrew! It's nice to meet you. How can I assist you today?
    Human: What is 1+1?
    AI: 1+1 equals 2. Is there anything else you would like to know?
    Human: What is my name?
    AI: Your name is Andrew.



```python
memory.load_memory_variables({})
```




    {'history': "Human: Hi, my name is Andrew\nAI: Hello Andrew! It's nice to meet you. How can I assist you today?\nHuman: What is 1+1?\nAI: 1+1 equals 2. Is there anything else you would like to know?\nHuman: What is my name?\nAI: Your name is Andrew."}




```python
memory = ConversationBufferMemory()
```


```python
memory.save_context({"input": "Hi"}, 
                    {"output": "What's up"})
```


```python
print(memory.buffer)
```

    Human: Hi
    AI: What's up



```python
memory.load_memory_variables({})
```




    {'history': "Human: Hi\nAI: What's up"}




```python
memory.save_context({"input": "Not much, just hanging"}, 
                    {"output": "Cool"})
```


```python
memory.load_memory_variables({})
```




    {'history': "Human: Hi\nAI: What's up\nHuman: Not much, just hanging\nAI: Cool"}



## ConversationBufferWindowMemory


```python
from langchain.memory import ConversationBufferWindowMemory
```


```python
memory = ConversationBufferWindowMemory(k=1)               
```


```python
memory.save_context({"input": "Hi"},
                    {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})

```


```python
memory.load_memory_variables({})
```




    {'history': 'Human: Not much, just hanging\nAI: Cool'}




```python
llm = ChatOpenAI(temperature=0.0, model=llm_model)
memory = ConversationBufferWindowMemory(k=1)
conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=False
)
```


```python
conversation.predict(input="Hi, my name is Andrew")
```




    "Hello Andrew! It's nice to meet you. How can I assist you today?"




```python
conversation.predict(input="What is 1+1?")
```




    '1+1 equals 2. Is there anything else you would like to know?'




```python
conversation.predict(input="What is my name?")
```




    "I'm sorry, I do not have access to personal information such as your name. Is there anything else you would like to know?"



## ConversationTokenBufferMemory


```python
#!pip install tiktoken
```


```python
from langchain.memory import ConversationTokenBufferMemory
from langchain.llms import OpenAI
llm = ChatOpenAI(temperature=0.0, model=llm_model)
```


```python
memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=50)
memory.save_context({"input": "AI is what?!"},
                    {"output": "Amazing!"})
memory.save_context({"input": "Backpropagation is what?"},
                    {"output": "Beautiful!"})
memory.save_context({"input": "Chatbots are what?"}, 
                    {"output": "Charming!"})
```


```python
memory.load_memory_variables({})
```




    {'history': 'AI: Amazing!\nHuman: Backpropagation is what?\nAI: Beautiful!\nHuman: Chatbots are what?\nAI: Charming!'}



## ConversationSummaryMemory


```python
from langchain.memory import ConversationSummaryBufferMemory
```


```python
# create a long string
schedule = "There is a meeting at 8am with your product team. \
You will need your powerpoint presentation prepared. \
9am-12pm have time to work on your LangChain \
project which will go quickly because Langchain is such a powerful tool. \
At Noon, lunch at the italian resturant with a customer who is driving \
from over an hour away to meet you to understand the latest in AI. \
Be sure to bring your laptop to show the latest LLM demo."

memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
memory.save_context({"input": "Hello"}, {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})
memory.save_context({"input": "What is on the schedule today?"}, 
                    {"output": f"{schedule}"})
```


```python
memory.load_memory_variables({})
```




    {'history': 'System: The human and AI exchange greetings and discuss the schedule for the day, including a meeting with the product team, work on the LangChain project, and a lunch meeting with a customer interested in AI. The AI provides details on each event and emphasizes the power of LangChain as a tool.'}




```python
conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=True
)
```


```python
conversation.predict(input="What would be a good demo to show?")
```

    
    
    [1m> Entering new ConversationChain chain...[0m
    Prompt after formatting:
    [32;1m[1;3mThe following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
    
    Current conversation:
    System: The human and AI exchange greetings and discuss the schedule for the day, including a meeting with the product team, work on the LangChain project, and a lunch meeting with a customer interested in AI. The AI provides details on each event and emphasizes the power of LangChain as a tool.
    Human: What would be a good demo to show?
    AI:[0m
    
    [1m> Finished chain.[0m





    "For the meeting with the product team, a demo showcasing the latest features and updates of LangChain would be ideal. This could include a live demonstration of how LangChain streamlines language processing tasks, improves accuracy, and increases efficiency in natural language understanding. Additionally, highlighting any recent success stories or case studies of LangChain's impact on businesses could be compelling for the audience."




```python
memory.load_memory_variables({})
```




    {'history': "System: The human and AI exchange greetings and discuss the schedule for the day, including a meeting with the product team, work on the LangChain project, and a lunch meeting with a customer interested in AI. The AI provides details on each event and emphasizes the power of LangChain as a tool.\nHuman: What would be a good demo to show?\nAI: For the meeting with the product team, a demo showcasing the latest features and updates of LangChain would be ideal. This could include a live demonstration of how LangChain streamlines language processing tasks, improves accuracy, and increases efficiency in natural language understanding. Additionally, highlighting any recent success stories or case studies of LangChain's impact on businesses could be compelling for the audience."}



Reminder: Download your notebook to you local computer to save your work.


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


```python

```


```python

```
