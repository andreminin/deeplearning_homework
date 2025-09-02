# Tools and Routing


```python
import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']
```


```python
from langchain.agents import tool
```


```python
@tool
def search(query: str) -> str:
    """Search for weather online"""
    return "42f"
```


```python
search.name
```




    'search'




```python
search.description
```




    'search(query: str) -> str - Search for weather online'




```python
search.args
```




    {'query': {'title': 'Query', 'type': 'string'}}




```python
from pydantic import BaseModel, Field
class SearchInput(BaseModel):
    query: str = Field(description="Thing to search for")

```


```python
@tool(args_schema=SearchInput)
def search(query: str) -> str:
    """Search for the weather online."""
    return "42f"
```


```python
search.args
```




    {'query': {'title': 'Query',
      'description': 'Thing to search for',
      'type': 'string'}}




```python
search.run("sf")
```




    '42f'




```python
import requests
from pydantic import BaseModel, Field
import datetime

# Define the input schema
class OpenMeteoInput(BaseModel):
    latitude: float = Field(..., description="Latitude of the location to fetch weather data for")
    longitude: float = Field(..., description="Longitude of the location to fetch weather data for")

@tool(args_schema=OpenMeteoInput)
def get_current_temperature(latitude: float, longitude: float) -> dict:
    """Fetch current temperature for given coordinates."""
    
    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    
    # Parameters for the request
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'hourly': 'temperature_2m',
        'forecast_days': 1,
    }

    # Make the request
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code == 200:
        results = response.json()
    else:
        raise Exception(f"API Request failed with status code: {response.status_code}")

    current_utc_time = datetime.datetime.utcnow()
    time_list = [datetime.datetime.fromisoformat(time_str.replace('Z', '+00:00')) for time_str in results['hourly']['time']]
    temperature_list = results['hourly']['temperature_2m']
    
    closest_time_index = min(range(len(time_list)), key=lambda i: abs(time_list[i] - current_utc_time))
    current_temperature = temperature_list[closest_time_index]
    
    return f'The current temperature is {current_temperature}°C'
```


```python
get_current_temperature.name
```




    'get_current_temperature'




```python
get_current_temperature.description
```




    'get_current_temperature(latitude: float, longitude: float) -> dict - Fetch current temperature for given coordinates.'




```python
get_current_temperature.args
```




    {'latitude': {'title': 'Latitude',
      'description': 'Latitude of the location to fetch weather data for',
      'type': 'number'},
     'longitude': {'title': 'Longitude',
      'description': 'Longitude of the location to fetch weather data for',
      'type': 'number'}}




```python
from langchain.tools.render import format_tool_to_openai_function
```


```python
format_tool_to_openai_function(get_current_temperature)
```




    {'name': 'get_current_temperature',
     'description': 'get_current_temperature(latitude: float, longitude: float) -> dict - Fetch current temperature for given coordinates.',
     'parameters': {'title': 'OpenMeteoInput',
      'type': 'object',
      'properties': {'latitude': {'title': 'Latitude',
        'description': 'Latitude of the location to fetch weather data for',
        'type': 'number'},
       'longitude': {'title': 'Longitude',
        'description': 'Longitude of the location to fetch weather data for',
        'type': 'number'}},
      'required': ['latitude', 'longitude']}}




```python
get_current_temperature({"latitude": 13, "longitude": 14})
```




    'The current temperature is 34.8°C'




```python
import wikipedia
@tool
def search_wikipedia(query: str) -> str:
    """Run Wikipedia search and get page summaries."""
    page_titles = wikipedia.search(query)
    summaries = []
    for page_title in page_titles[: 3]:
        try:
            wiki_page =  wikipedia.page(title=page_title, auto_suggest=False)
            summaries.append(f"Page: {page_title}\nSummary: {wiki_page.summary}")
        except (
            self.wiki_client.exceptions.PageError,
            self.wiki_client.exceptions.DisambiguationError,
        ):
            pass
    if not summaries:
        return "No good Wikipedia Search Result was found"
    return "\n\n".join(summaries)
```


```python
search_wikipedia.name
```




    'search_wikipedia'




```python
search_wikipedia.description
```




    'search_wikipedia(query: str) -> str - Run Wikipedia search and get page summaries.'




```python
format_tool_to_openai_function(search_wikipedia)
```




    {'name': 'search_wikipedia',
     'description': 'search_wikipedia(query: str) -> str - Run Wikipedia search and get page summaries.',
     'parameters': {'title': 'search_wikipediaSchemaSchema',
      'type': 'object',
      'properties': {'query': {'title': 'Query', 'type': 'string'}},
      'required': ['query']}}




```python
search_wikipedia({"query": "langchain"})
```




    'Page: LangChain\nSummary: LangChain is a software framework that helps facilitate the integration of large language models (LLMs) into applications. As a language model integration framework, LangChain\'s use-cases largely overlap with those of language models in general, including document analysis and summarization, chatbots, and code analysis.\n\n\n\nPage: Model Context Protocol\nSummary: The Model Context Protocol (MCP) is an open standard, open-source framework introduced by Anthropic in November 2024 to standardize the way artificial intelligence (AI) systems like large language models (LLMs) integrate and share data with external tools, systems, and data sources. MCP provides a universal interface for reading files, executing functions, and handling contextual prompts. Following its announcement, the protocol was adopted by major AI providers, including OpenAI and Google DeepMind.\n\nPage: Retrieval-augmented generation\nSummary: Retrieval-augmented generation (RAG) is a technique that enables large language models (LLMs) to retrieve and incorporate new information. With RAG, LLMs do not respond to user queries until they refer to a specified set of documents. These documents supplement information from the LLM\'s pre-existing training data. This allows LLMs to use domain-specific and/or updated information that is not available in the training data. For example, this helps LLM-based chatbots access internal company data or generate responses based on authoritative sources.\nRAG improves large language models (LLMs) by incorporating information retrieval before generating responses. Unlike traditional LLMs that rely on static training data, RAG pulls relevant text from databases, uploaded documents, or web sources. According to Ars Technica, "RAG is a way of improving LLM performance, in essence by blending the LLM process with a web search or other document look-up process to help LLMs stick to the facts." This method helps reduce AI hallucinations, which have caused chatbots to describe policies that don\'t exist, or recommend nonexistent legal cases to lawyers that are looking for citations to support their arguments.\nRAG also reduces the need to retrain LLMs with new data, saving on computational and financial costs. Beyond efficiency gains, RAG also allows LLMs to include sources in their responses, so users can verify the cited sources. This provides greater transparency, as users can cross-check retrieved content to ensure accuracy and relevance.\nThe term RAG was first introduced in a 2020 research paper from Meta.'




```python
from langchain.chains.openai_functions.openapi import openapi_spec_to_openai_fn
from langchain.utilities.openapi import OpenAPISpec
```


```python
text = """
{
  "openapi": "3.1.0",
  "info": {
    "version": "1.0.0",
    "title": "Swagger Petstore",
    "license": {
      "name": "MIT"
    }
  },
  "servers": [
    {
      "url": "http://petstore.swagger.io/v1"
    }
  ],
  "paths": {
    "/pets": {
      "get": {
        "summary": "List all pets",
        "operationId": "listPets",
        "tags": [
          "pets"
        ],
        "parameters": [
          {
            "name": "limit",
            "in": "query",
            "description": "How many items to return at one time (max 100)",
            "required": false,
            "schema": {
              "type": "integer",
              "maximum": 100,
              "format": "int32"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "A paged array of pets",
            "headers": {
              "x-next": {
                "description": "A link to the next page of responses",
                "schema": {
                  "type": "string"
                }
              }
            },
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Pets"
                }
              }
            }
          },
          "default": {
            "description": "unexpected error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        }
      },
      "post": {
        "summary": "Create a pet",
        "operationId": "createPets",
        "tags": [
          "pets"
        ],
        "responses": {
          "201": {
            "description": "Null response"
          },
          "default": {
            "description": "unexpected error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        }
      }
    },
    "/pets/{petId}": {
      "get": {
        "summary": "Info for a specific pet",
        "operationId": "showPetById",
        "tags": [
          "pets"
        ],
        "parameters": [
          {
            "name": "petId",
            "in": "path",
            "required": true,
            "description": "The id of the pet to retrieve",
            "schema": {
              "type": "string"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Expected response to a valid request",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Pet"
                }
              }
            }
          },
          "default": {
            "description": "unexpected error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        }
      }
    }
  },
  "components": {
    "schemas": {
      "Pet": {
        "type": "object",
        "required": [
          "id",
          "name"
        ],
        "properties": {
          "id": {
            "type": "integer",
            "format": "int64"
          },
          "name": {
            "type": "string"
          },
          "tag": {
            "type": "string"
          }
        }
      },
      "Pets": {
        "type": "array",
        "maxItems": 100,
        "items": {
          "$ref": "#/components/schemas/Pet"
        }
      },
      "Error": {
        "type": "object",
        "required": [
          "code",
          "message"
        ],
        "properties": {
          "code": {
            "type": "integer",
            "format": "int32"
          },
          "message": {
            "type": "string"
          }
        }
      }
    }
  }
}
"""
```


```python
spec = OpenAPISpec.from_text(text)
```


```python
pet_openai_functions, pet_callables = openapi_spec_to_openai_fn(spec)
```


```python
pet_openai_functions
```




    [{'name': 'listPets',
      'description': 'List all pets',
      'parameters': {'type': 'object',
       'properties': {'params': {'type': 'object',
         'properties': {'limit': {'type': 'integer',
           'maximum': 100.0,
           'schema_format': 'int32',
           'description': 'How many items to return at one time (max 100)'}},
         'required': []}}}},
     {'name': 'createPets',
      'description': 'Create a pet',
      'parameters': {'type': 'object', 'properties': {}}},
     {'name': 'showPetById',
      'description': 'Info for a specific pet',
      'parameters': {'type': 'object',
       'properties': {'path_params': {'type': 'object',
         'properties': {'petId': {'type': 'string',
           'description': 'The id of the pet to retrieve'}},
         'required': ['petId']}}}}]




```python
from langchain.chat_models import ChatOpenAI
```


```python
model = ChatOpenAI(temperature=0).bind(functions=pet_openai_functions)
```


```python
model.invoke("what are three pets names")
```




    AIMessage(content='', additional_kwargs={'function_call': {'name': 'listPets', 'arguments': '{"params":{"limit":3}}'}})




```python
model.invoke("tell me about pet with id 42")
```




    AIMessage(content='', additional_kwargs={'function_call': {'name': 'showPetById', 'arguments': '{"path_params":{"petId":"42"}}'}})



### Routing

In lesson 3, we show an example of function calling deciding between two candidate functions.

Given our tools above, let's format these as OpenAI functions and show this same behavior.


```python
functions = [
    format_tool_to_openai_function(f) for f in [
        search_wikipedia, get_current_temperature
    ]
]
model = ChatOpenAI(temperature=0).bind(functions=functions)
```


```python
model.invoke("what is the weather in sf right now")
```




    AIMessage(content='', additional_kwargs={'function_call': {'name': 'get_current_temperature', 'arguments': '{"latitude":37.7749,"longitude":-122.4194}'}})




```python
model.invoke("what is langchain")
```




    AIMessage(content='', additional_kwargs={'function_call': {'name': 'search_wikipedia', 'arguments': '{"query":"Langchain"}'}})




```python
from langchain.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful but sassy assistant"),
    ("user", "{input}"),
])
chain = prompt | model
```


```python
chain.invoke({"input": "what is the weather in sf right now"})
```




    AIMessage(content='', additional_kwargs={'function_call': {'name': 'get_current_temperature', 'arguments': '{"latitude":37.7749,"longitude":-122.4194}'}})




```python
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
```


```python
chain = prompt | model | OpenAIFunctionsAgentOutputParser()
```


```python
result = chain.invoke({"input": "what is the weather in sf right now"})
```


```python
type(result)
```




    langchain.schema.agent.AgentActionMessageLog




```python
result.tool
```




    'get_current_temperature'




```python
result.tool_input
```




    {'latitude': 37.7749, 'longitude': -122.4194}




```python
get_current_temperature(result.tool_input)
```




    'The current temperature is 13.3°C'




```python
result = chain.invoke({"input": "hi!"})
```


```python
type(result)
```




    langchain.schema.agent.AgentFinish




```python
result.return_values
```




    {'output': 'Hello! How can I assist you today?'}




```python
from langchain.schema.agent import AgentFinish
def route(result):
    if isinstance(result, AgentFinish):
        return result.return_values['output']
    else:
        tools = {
            "search_wikipedia": search_wikipedia, 
            "get_current_temperature": get_current_temperature,
        }
        return tools[result.tool].run(result.tool_input)
```


```python
chain = prompt | model | OpenAIFunctionsAgentOutputParser() | route
```


```python
result = chain.invoke({"input": "What is the weather in san francisco right now?"})
```


```python
result
```




    'The current temperature is 13.3°C'




```python
result = chain.invoke({"input": "What is langchain?"})
```


```python
result
```




    'Page: LangChain\nSummary: LangChain is a software framework that helps facilitate the integration of large language models (LLMs) into applications. As a language model integration framework, LangChain\'s use-cases largely overlap with those of language models in general, including document analysis and summarization, chatbots, and code analysis.\n\n\n\nPage: Retrieval-augmented generation\nSummary: Retrieval-augmented generation (RAG) is a technique that enables large language models (LLMs) to retrieve and incorporate new information. With RAG, LLMs do not respond to user queries until they refer to a specified set of documents. These documents supplement information from the LLM\'s pre-existing training data. This allows LLMs to use domain-specific and/or updated information that is not available in the training data. For example, this helps LLM-based chatbots access internal company data or generate responses based on authoritative sources.\nRAG improves large language models (LLMs) by incorporating information retrieval before generating responses. Unlike traditional LLMs that rely on static training data, RAG pulls relevant text from databases, uploaded documents, or web sources. According to Ars Technica, "RAG is a way of improving LLM performance, in essence by blending the LLM process with a web search or other document look-up process to help LLMs stick to the facts." This method helps reduce AI hallucinations, which have caused chatbots to describe policies that don\'t exist, or recommend nonexistent legal cases to lawyers that are looking for citations to support their arguments.\nRAG also reduces the need to retrain LLMs with new data, saving on computational and financial costs. Beyond efficiency gains, RAG also allows LLMs to include sources in their responses, so users can verify the cited sources. This provides greater transparency, as users can cross-check retrieved content to ensure accuracy and relevance.\nThe term RAG was first introduced in a 2020 research paper from Meta.\n\nPage: Model Context Protocol\nSummary: The Model Context Protocol (MCP) is an open standard, open-source framework introduced by Anthropic in November 2024 to standardize the way artificial intelligence (AI) systems like large language models (LLMs) integrate and share data with external tools, systems, and data sources. MCP provides a universal interface for reading files, executing functions, and handling contextual prompts. Following its announcement, the protocol was adopted by major AI providers, including OpenAI and Google DeepMind.'




```python
chain.invoke({"input": "hi!"})
```




    'Hello! How can I assist you today?'




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
