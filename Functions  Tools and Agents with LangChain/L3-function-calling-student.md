# OpenAI Function Calling In LangChain


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
```

## Pydantic Syntax

Pydantic data classes are a blend of Python's data classes with the validation power of Pydantic. 

They offer a concise way to define data structures while ensuring that the data adheres to specified types and constraints.

In standard python you would create a class like this:


```python
class User:
    def __init__(self, name: str, age: int, email: str):
        self.name = name
        self.age = age
        self.email = email
```


```python
foo = User(name="Joe",age=32, email="joe@gmail.com")
```


```python
foo.name
```




    'Joe'




```python
foo = User(name="Joe",age="bar", email="joe@gmail.com")
```


```python
foo.age
```




    'bar'




```python
class pUser(BaseModel):
    name: str
    age: int
    email: str
```


```python
foo_p = pUser(name="Jane", age=32, email="jane@gmail.com")
```


```python
foo_p.name
```




    'Jane'




```python
foo_p.age
```




    32



<p style=\"background-color:#F5C780; padding:15px\"><b>Note:</b> The next line is expected to fail.</p>


```python
foo_p = pUser(name="Jane", age="bar", email="jane@gmail.com")
```


    ---------------------------------------------------------------------------

    ValidationError                           Traceback (most recent call last)

    Cell In[14], line 1
    ----> 1 foo_p = pUser(name="Jane", age="bar", email="jane@gmail.com")


    File /usr/local/lib/python3.9/site-packages/pydantic/main.py:341, in pydantic.main.BaseModel.__init__()


    ValidationError: 1 validation error for pUser
    age
      value is not a valid integer (type=type_error.integer)



```python
class Class(BaseModel):
    students: List[pUser]
```


```python
obj = Class(
    students=[pUser(name="Jane", age=32, email="jane@gmail.com")]
)
```


```python
obj
```




    Class(students=[pUser(name='Jane', age=32, email='jane@gmail.com')])



## Pydantic to OpenAI function definition



```python
class WeatherSearch(BaseModel):
    """Call this with an airport code to get the weather at that airport"""
    airport_code: str = Field(description="airport code to get weather for")
```


```python
 
```


```python
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
```


```python
weather_function = convert_pydantic_to_openai_function(WeatherSearch)
```


```python
weather_function
```




    {'name': 'WeatherSearch',
     'description': 'Call this with an airport code to get the weather at that airport',
     'parameters': {'title': 'WeatherSearch',
      'description': 'Call this with an airport code to get the weather at that airport',
      'type': 'object',
      'properties': {'airport_code': {'title': 'Airport Code',
        'description': 'airport code to get weather for',
        'type': 'string'}},
      'required': ['airport_code']}}




```python
class WeatherSearch1(BaseModel):
    airport_code: str = Field(description="airport code to get weather for")
```

<p style=\"background-color:#F5C780; padding:15px\"><b>Note:</b> The next cell is expected to generate an error.</p>


```python
convert_pydantic_to_openai_function(WeatherSearch1)
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    Cell In[29], line 1
    ----> 1 convert_pydantic_to_openai_function(WeatherSearch1)


    File /usr/local/lib/python3.9/site-packages/langchain/utils/openai_functions.py:28, in convert_pydantic_to_openai_function(model, name, description)
         24 schema = dereference_refs(model.schema())
         25 schema.pop("definitions", None)
         26 return {
         27     "name": name or schema["title"],
    ---> 28     "description": description or schema["description"],
         29     "parameters": schema,
         30 }


    KeyError: 'description'



```python
class WeatherSearch2(BaseModel):
    """Call this with an airport code to get the weather at that airport"""
    airport_code: str
```


```python
convert_pydantic_to_openai_function(WeatherSearch2)
```




    {'name': 'WeatherSearch2',
     'description': 'Call this with an airport code to get the weather at that airport',
     'parameters': {'title': 'WeatherSearch2',
      'description': 'Call this with an airport code to get the weather at that airport',
      'type': 'object',
      'properties': {'airport_code': {'title': 'Airport Code', 'type': 'string'}},
      'required': ['airport_code']}}




```python
from langchain.chat_models import ChatOpenAI
```


```python
model = ChatOpenAI()
```


```python
model.invoke("what is the weather in SF today?", functions=[weather_function])
```




    AIMessage(content='', additional_kwargs={'function_call': {'name': 'WeatherSearch', 'arguments': '{"airport_code":"SFO"}'}})




```python
model_with_function = model.bind(functions=[weather_function])
```


```python
model_with_function.invoke("what is the weather in sf?")
```




    AIMessage(content='', additional_kwargs={'function_call': {'name': 'WeatherSearch', 'arguments': '{"airport_code":"SFO"}'}})



## Forcing it to use a function

We can force the model to use a function


```python
model_with_forced_function = model.bind(functions=[weather_function], function_call={"name":"WeatherSearch"})
```


```python
model_with_forced_function.invoke("what is the weather in sf?")
```




    AIMessage(content='', additional_kwargs={'function_call': {'name': 'WeatherSearch', 'arguments': '{"airport_code":"SFO"}'}})




```python
model_with_forced_function.invoke("hi!")
```




    AIMessage(content='', additional_kwargs={'function_call': {'name': 'WeatherSearch', 'arguments': '{"airport_code":"JFK"}'}})



## Using in a chain

We can use this model bound to function in a chain as we normally would


```python
from langchain.prompts import ChatPromptTemplate
```


```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("user", "{input}")
])
```


```python
chain = prompt | model_with_function
```


```python
chain.invoke({"input": "what is the weather in sf?"})
```




    AIMessage(content='', additional_kwargs={'function_call': {'name': 'WeatherSearch', 'arguments': '{"airport_code":"SFO"}'}})



## Using multiple functions

Even better, we can pass a set of function and let the LLM decide which to use based on the question context.


```python
class ArtistSearch(BaseModel):
    """Call this to get the names of songs by a particular artist"""
    artist_name: str = Field(description="name of artist to look up")
    n: int = Field(description="number of results")
```


```python
functions = [
    convert_pydantic_to_openai_function(WeatherSearch),
    convert_pydantic_to_openai_function(ArtistSearch),
]
```


```python
model_with_functions = model.bind(functions=functions)
```


```python
model_with_functions.invoke("what is the weather in sf?")
```




    AIMessage(content='', additional_kwargs={'function_call': {'name': 'WeatherSearch', 'arguments': '{"airport_code":"SFO"}'}})




```python
model_with_functions.invoke("what are three songs by taylor swift?")
```




    AIMessage(content='', additional_kwargs={'function_call': {'name': 'ArtistSearch', 'arguments': '{"artist_name":"Taylor Swift","n":3}'}})




```python
model_with_functions.invoke("hi!")
```




    AIMessage(content='Hello! How can I assist you today?')




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
