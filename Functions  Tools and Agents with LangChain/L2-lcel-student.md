# LangChain Expression Language (LCEL)


```python
import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']
```


```python
#!pip install pydantic==1.10.8
```


```python
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
```

## Simple Chain


```python
prompt = ChatPromptTemplate.from_template(
    "tell me a short joke about {topic}"
)
model = ChatOpenAI()
output_parser = StrOutputParser()
```


```python
chain = prompt | model | output_parser
```


```python
chain.invoke({"topic": "bears"})
```




    'Why did the bear bring a flashlight to the party? \n\nBecause he heard it was going to be a "beary" good time!'



## More complex chain

And Runnable Map to supply user-provided inputs to the prompt.


```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
```


```python
vectorstore = DocArrayInMemorySearch.from_texts(
    ["harrison worked at kensho", "bears like to eat honey"],
    embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()
```


```python
retriever.get_relevant_documents("where did harrison work?")
```




    [Document(page_content='harrison worked at kensho'),
     Document(page_content='bears like to eat honey')]




```python
retriever.get_relevant_documents("what do bears like to eat")
```




    [Document(page_content='bears like to eat honey'),
     Document(page_content='harrison worked at kensho')]




```python
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
```


```python
from langchain.schema.runnable import RunnableMap
```


```python
chain = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | model | output_parser
```


```python
chain.invoke({"question": "where did harrison work?"})
```




    'Harrison worked at Kensho.'




```python
inputs = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
})
```


```python
inputs.invoke({"question": "where did harrison work?"})
```




    {'context': [Document(page_content='harrison worked at kensho'),
      Document(page_content='bears like to eat honey')],
     'question': 'where did harrison work?'}



## Bind

and OpenAI Functions


```python
functions = [
    {
      "name": "weather_search",
      "description": "Search for weather given an airport code",
      "parameters": {
        "type": "object",
        "properties": {
          "airport_code": {
            "type": "string",
            "description": "The airport code to get the weather for"
          },
        },
        "required": ["airport_code"]
      }
    }
  ]
```


```python
prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}")
    ]
)
model = ChatOpenAI(temperature=0).bind(functions=functions)
```


```python
runnable = prompt | model
```


```python
runnable.invoke({"input": "what is the weather in sf"})
```




    AIMessage(content='', additional_kwargs={'function_call': {'name': 'weather_search', 'arguments': '{"airport_code":"SFO"}'}})




```python
functions = [
    {
      "name": "weather_search",
      "description": "Search for weather given an airport code",
      "parameters": {
        "type": "object",
        "properties": {
          "airport_code": {
            "type": "string",
            "description": "The airport code to get the weather for"
          },
        },
        "required": ["airport_code"]
      }
    },
        {
      "name": "sports_search",
      "description": "Search for news of recent sport events",
      "parameters": {
        "type": "object",
        "properties": {
          "team_name": {
            "type": "string",
            "description": "The sports team to search for"
          },
        },
        "required": ["team_name"]
      }
    }
  ]
```


```python
model = model.bind(functions=functions)
```


```python
runnable = prompt | model
```


```python
runnable.invoke({"input": "how did the patriots do yesterday?"})
```




    AIMessage(content='', additional_kwargs={'function_call': {'name': 'sports_search', 'arguments': '{"team_name":"New England Patriots"}'}})



## Fallbacks


```python
from langchain.llms import OpenAI
import json
```

**Note**: Due to the deprication of OpenAI's model `text-davinci-001` on 4 January 2024, you'll be using OpenAI's recommended replacement model `gpt-3.5-turbo-instruct` instead.


```python
simple_model = OpenAI(
    temperature=0, 
    max_tokens=1000, 
    model="gpt-3.5-turbo-instruct"
)
simple_chain = simple_model | json.loads
```


```python
challenge = "write three poems in a json blob, where each poem is a json blob of a title, author, and first line"
```


```python
simple_model.invoke(challenge)
```




    '\n\n{\n    "title": "Autumn Leaves",\n    "author": "Emily Dickinson",\n    "first_line": "The leaves are falling, one by one"\n}\n\n{\n    "title": "The Ocean\'s Song",\n    "author": "Pablo Neruda",\n    "first_line": "I hear the ocean\'s song, a symphony of waves"\n}\n\n{\n    "title": "A Winter\'s Night",\n    "author": "Robert Frost",\n    "first_line": "The snow falls softly, covering the ground"\n}'



<p style=\"background-color:#F5C780; padding:15px\"><b>Note:</b> The next line is expected to fail.</p>


```python
simple_chain.invoke(challenge)
```


    ---------------------------------------------------------------------------

    JSONDecodeError                           Traceback (most recent call last)

    Cell In[29], line 1
    ----> 1 simple_chain.invoke(challenge)


    File /usr/local/lib/python3.9/site-packages/langchain/schema/runnable/base.py:1113, in RunnableSequence.invoke(self, input, config)
       1111 try:
       1112     for i, step in enumerate(self.steps):
    -> 1113         input = step.invoke(
       1114             input,
       1115             # mark each step as a child run
       1116             patch_config(
       1117                 config, callbacks=run_manager.get_child(f"seq:step:{i+1}")
       1118             ),
       1119         )
       1120 # finish the root run
       1121 except BaseException as e:


    File /usr/local/lib/python3.9/site-packages/langchain/schema/runnable/base.py:2163, in RunnableLambda.invoke(self, input, config, **kwargs)
       2161 """Invoke this runnable synchronously."""
       2162 if hasattr(self, "func"):
    -> 2163     return self._call_with_config(
       2164         self._invoke,
       2165         input,
       2166         self._config(config, self.func),
       2167     )
       2168 else:
       2169     raise TypeError(
       2170         "Cannot invoke a coroutine function synchronously."
       2171         "Use `ainvoke` instead."
       2172     )


    File /usr/local/lib/python3.9/site-packages/langchain/schema/runnable/base.py:633, in Runnable._call_with_config(self, func, input, config, run_type, **kwargs)
        626 run_manager = callback_manager.on_chain_start(
        627     dumpd(self),
        628     input,
        629     run_type=run_type,
        630     name=config.get("run_name"),
        631 )
        632 try:
    --> 633     output = call_func_with_variable_args(
        634         func, input, run_manager, config, **kwargs
        635     )
        636 except BaseException as e:
        637     run_manager.on_chain_error(e)


    File /usr/local/lib/python3.9/site-packages/langchain/schema/runnable/config.py:173, in call_func_with_variable_args(func, input, run_manager, config, **kwargs)
        171 if accepts_run_manager(func):
        172     kwargs["run_manager"] = run_manager
    --> 173 return func(input, **kwargs)


    File /usr/local/lib/python3.9/site-packages/langchain/schema/runnable/base.py:2096, in RunnableLambda._invoke(self, input, run_manager, config)
       2090 def _invoke(
       2091     self,
       2092     input: Input,
       2093     run_manager: CallbackManagerForChainRun,
       2094     config: RunnableConfig,
       2095 ) -> Output:
    -> 2096     output = call_func_with_variable_args(self.func, input, run_manager, config)
       2097     # If the output is a runnable, invoke it
       2098     if isinstance(output, Runnable):


    File /usr/local/lib/python3.9/site-packages/langchain/schema/runnable/config.py:173, in call_func_with_variable_args(func, input, run_manager, config, **kwargs)
        171 if accepts_run_manager(func):
        172     kwargs["run_manager"] = run_manager
    --> 173 return func(input, **kwargs)


    File /usr/local/lib/python3.9/json/__init__.py:346, in loads(s, cls, object_hook, parse_float, parse_int, parse_constant, object_pairs_hook, **kw)
        341     s = s.decode(detect_encoding(s), 'surrogatepass')
        343 if (cls is None and object_hook is None and
        344         parse_int is None and parse_float is None and
        345         parse_constant is None and object_pairs_hook is None and not kw):
    --> 346     return _default_decoder.decode(s)
        347 if cls is None:
        348     cls = JSONDecoder


    File /usr/local/lib/python3.9/json/decoder.py:340, in JSONDecoder.decode(self, s, _w)
        338 end = _w(s, end).end()
        339 if end != len(s):
    --> 340     raise JSONDecodeError("Extra data", s, end)
        341 return obj


    JSONDecodeError: Extra data: line 9 column 1 (char 125)



```python
model = ChatOpenAI(temperature=0)
chain = model | StrOutputParser() | json.loads
```


```python
chain.invoke(challenge)
```




    {'poem1': {'title': 'The Rose',
      'author': 'Emily Dickinson',
      'firstLine': 'A rose by any other name would smell as sweet'},
     'poem2': {'title': 'The Road Not Taken',
      'author': 'Robert Frost',
      'firstLine': 'Two roads diverged in a yellow wood'},
     'poem3': {'title': 'Hope is the Thing with Feathers',
      'author': 'Emily Dickinson',
      'firstLine': 'Hope is the thing with feathers that perches in the soul'}}




```python
final_chain = simple_chain.with_fallbacks([chain])
```


```python
final_chain.invoke(challenge)
```




    {'poem1': {'title': 'The Rose',
      'author': 'Emily Dickinson',
      'firstLine': 'A rose by any other name would smell as sweet'},
     'poem2': {'title': 'The Road Not Taken',
      'author': 'Robert Frost',
      'firstLine': 'Two roads diverged in a yellow wood'},
     'poem3': {'title': 'Hope is the Thing with Feathers',
      'author': 'Emily Dickinson',
      'firstLine': 'Hope is the thing with feathers that perches in the soul'}}



## Interface


```python
prompt = ChatPromptTemplate.from_template(
    "Tell me a short joke about {topic}"
)
model = ChatOpenAI()
output_parser = StrOutputParser()

chain = prompt | model | output_parser
```


```python
chain.invoke({"topic": "bears"})
```




    "Why do bears have hairy coats?\n\nBecause they don't like to shave!"




```python
chain.batch([{"topic": "bears"}, {"topic": "frogs"}])
```




    ["Why do bears have hairy coats?\n\nBecause they don't like to shave!",
     'Why are frogs so happy?\nBecause they eat whatever bugs them!']




```python
for t in chain.stream({"topic": "bears"}):
    print(t)
```

    
    Why
     did
     the
     bear
     break
     up
     with
     his
     girlfriend
    ?
     
    
    
    Because
     he
     couldn
    't
     bear
     the
     relationship
     anymore
    !
    



```python
response = await chain.ainvoke({"topic": "bears"})
response
```




    "Why do bears have hairy coats?\n\nBecause they don't like to shave!"




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
