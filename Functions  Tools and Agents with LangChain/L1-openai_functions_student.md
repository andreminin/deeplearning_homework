# OpenAI Function Calling


**Notes**:
- LLMs don't always produce the same results. The results you see in this notebook may differ from the results you see in the video.
- Notebooks results are temporary. Download the notebooks to your local machine if you wish to save your results.
- OpenAI has announced the release of an updated GPT-3.5-Turbo model and the deprecation of the ```gpt-3.5-turbo-0613``` model. The gpt-3.5-turbo-0613 model, which has been utilized in this Short Course since its launch in October 2023, will be replaced by the ```gpt-3.5-turbo``` model.


```python
import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']
```


```python
import json

# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    weather_info = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)
```


```python
# define a function
functions = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    }
]
```


```python
messages = [
    {
        "role": "user",
        "content": "What's the weather like in Boston?"
    }
]
```


```python
import openai
```

> Note: This notebook was updated in June 2024. Consequently, we are now using the ```gpt-3.5-turbo model``` instead of the ```gpt-3.5-turbo-0613``` model featured by the instructor in the video lesson.


```python
# Call the ChatCompletion endpoint
response = openai.ChatCompletion.create(
    # OpenAI Updates: As of June 2024, we are now using the GPT-3.5-Turbo model
    model="gpt-3.5-turbo",
    messages=messages,
    functions=functions
)
```

> Note: The following result may differ slightly from the one shown by the instructor in the video lesson due to the model being updated.


```python
print(response)
```

    {
      "id": "chatcmpl-AVCsmqFCt4FILgYpy7wDxpjahOZvC",
      "object": "chat.completion",
      "created": 1732001052,
      "model": "gpt-3.5-turbo-0125",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": null,
            "function_call": {
              "name": "get_current_weather",
              "arguments": "{\"location\":\"Boston\",\"unit\":\"celsius\"}"
            },
            "refusal": null
          },
          "logprobs": null,
          "finish_reason": "function_call"
        }
      ],
      "usage": {
        "prompt_tokens": 82,
        "completion_tokens": 20,
        "total_tokens": 102,
        "prompt_tokens_details": {
          "cached_tokens": 0,
          "audio_tokens": 0
        },
        "completion_tokens_details": {
          "reasoning_tokens": 0,
          "audio_tokens": 0,
          "accepted_prediction_tokens": 0,
          "rejected_prediction_tokens": 0
        }
      },
      "system_fingerprint": null
    }



```python
response_message = response["choices"][0]["message"]
```


```python
response_message
```




    <OpenAIObject at 0x7f653801cd10> JSON: {
      "role": "assistant",
      "content": null,
      "function_call": {
        "name": "get_current_weather",
        "arguments": "{\"location\":\"Boston\",\"unit\":\"celsius\"}"
      },
      "refusal": null
    }




```python
response_message["content"]
```


```python
response_message["function_call"]
```




    <OpenAIObject at 0x7f653801cd60> JSON: {
      "name": "get_current_weather",
      "arguments": "{\"location\":\"Boston\",\"unit\":\"celsius\"}"
    }




```python
json.loads(response_message["function_call"]["arguments"])
```




    {'location': 'Boston', 'unit': 'celsius'}




```python
args = json.loads(response_message["function_call"]["arguments"])
```


```python
get_current_weather(args)
```




    '{"location": {"location": "Boston", "unit": "celsius"}, "temperature": "72", "unit": "fahrenheit", "forecast": ["sunny", "windy"]}'



* Pass a message that is not related to a function.


```python
messages = [
    {
        "role": "user",
        "content": "hi!",
    }
]
```

> Note: This notebook was updated in June 2024. Consequently, we are now using the ```gpt-3.5-turbo model``` instead of the ```gpt-3.5-turbo-0613``` model featured by the instructor in the video lesson.


```python
response = openai.ChatCompletion.create(
    # OpenAI Updates: As of June 2024, we are now using the GPT-3.5-Turbo model
    model="gpt-3.5-turbo",
    messages=messages,
    functions=functions,
)
```


```python
print(response)
```

    {
      "id": "chatcmpl-BU7Es7EKBrxor2FTiWqtZnDzQPH5X",
      "object": "chat.completion",
      "created": 1746517366,
      "model": "gpt-3.5-turbo-0125",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": "Hello! How can I assist you today?",
            "refusal": null,
            "annotations": []
          },
          "logprobs": null,
          "finish_reason": "stop"
        }
      ],
      "usage": {
        "prompt_tokens": 76,
        "completion_tokens": 11,
        "total_tokens": 87,
        "prompt_tokens_details": {
          "cached_tokens": 0,
          "audio_tokens": 0
        },
        "completion_tokens_details": {
          "reasoning_tokens": 0,
          "audio_tokens": 0,
          "accepted_prediction_tokens": 0,
          "rejected_prediction_tokens": 0
        }
      },
      "service_tier": "default",
      "system_fingerprint": null
    }


* Pass additional parameters to force the model to use or not a function.


```python
messages = [
    {
        "role": "user",
        "content": "hi!",
    }
]
response = openai.ChatCompletion.create(
    # OpenAI Updates: As of June 2024, we are now using the GPT-3.5-Turbo model
    model="gpt-3.5-turbo",
    messages=messages,
    functions=functions,
    function_call="auto",
)
print(response)
```

    {
      "id": "chatcmpl-B8QrXbJYGHGRiybD6m860Rw7mzhFH",
      "object": "chat.completion",
      "created": 1741349583,
      "model": "gpt-3.5-turbo-0125",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": "Hello! How can I assist you today?",
            "refusal": null
          },
          "logprobs": null,
          "finish_reason": "stop"
        }
      ],
      "usage": {
        "prompt_tokens": 76,
        "completion_tokens": 11,
        "total_tokens": 87,
        "prompt_tokens_details": {
          "cached_tokens": 0,
          "audio_tokens": 0
        },
        "completion_tokens_details": {
          "reasoning_tokens": 0,
          "audio_tokens": 0,
          "accepted_prediction_tokens": 0,
          "rejected_prediction_tokens": 0
        }
      },
      "service_tier": "default",
      "system_fingerprint": null
    }


* Use mode 'none' for function call.


```python
messages = [
    {
        "role": "user",
        "content": "hi!",
    }
]
response = openai.ChatCompletion.create(
    # OpenAI Updates: As of June 2024, we are now using the GPT-3.5-Turbo model
    model="gpt-3.5-turbo",
    messages=messages,
    functions=functions,
    function_call="none",
)
print(response)
```

    {
      "id": "chatcmpl-AVD6QpNW55BDUQ2E1bXyaskKynDKb",
      "object": "chat.completion",
      "created": 1732001898,
      "model": "gpt-3.5-turbo-0125",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": "Hello! How can I assist you today?",
            "refusal": null
          },
          "logprobs": null,
          "finish_reason": "stop"
        }
      ],
      "usage": {
        "prompt_tokens": 77,
        "completion_tokens": 9,
        "total_tokens": 86,
        "prompt_tokens_details": {
          "cached_tokens": 0,
          "audio_tokens": 0
        },
        "completion_tokens_details": {
          "reasoning_tokens": 0,
          "audio_tokens": 0,
          "accepted_prediction_tokens": 0,
          "rejected_prediction_tokens": 0
        }
      },
      "system_fingerprint": null
    }


* When the message should call a function and still uses mode 'none'.


```python
messages = [
    {
        "role": "user",
        "content": "What's the weather in Boston?",
    }
]
response = openai.ChatCompletion.create(
    # OpenAI Updates: As of June 2024, we are now using the GPT-3.5-Turbo model
    model="gpt-3.5-turbo",
    messages=messages,
    functions=functions,
    function_call="none",
)
print(response)
```

    {
      "id": "chatcmpl-AVD7PzmAAhjBqgvw8V7vGP8A2k7D5",
      "object": "chat.completion",
      "created": 1732001959,
      "model": "gpt-3.5-turbo-0125",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": "I can find out the weather in Boston for you. Should I display the temperature in Celsius or Fahrenheit?",
            "refusal": null
          },
          "logprobs": null,
          "finish_reason": "stop"
        }
      ],
      "usage": {
        "prompt_tokens": 82,
        "completion_tokens": 21,
        "total_tokens": 103,
        "prompt_tokens_details": {
          "cached_tokens": 0,
          "audio_tokens": 0
        },
        "completion_tokens_details": {
          "reasoning_tokens": 0,
          "audio_tokens": 0,
          "accepted_prediction_tokens": 0,
          "rejected_prediction_tokens": 0
        }
      },
      "system_fingerprint": null
    }


* Force calling a function.


```python
messages = [
    {
        "role": "user",
        "content": "hi!",
    }
]
response = openai.ChatCompletion.create(
    # OpenAI Updates: As of June 2024, we are now using the GPT-3.5-Turbo model
    model="gpt-3.5-turbo",
    messages=messages,
    functions=functions,
    function_call={"name": "get_current_weather"},
)
print(response)
```

    {
      "id": "chatcmpl-BUsxEG1eTCxlzbVhINBuHe30anTlZ",
      "object": "chat.completion",
      "created": 1746700784,
      "model": "gpt-3.5-turbo-0125",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": null,
            "function_call": {
              "name": "get_current_weather",
              "arguments": "{\"location\":\"San Francisco, CA\"}"
            },
            "refusal": null,
            "annotations": []
          },
          "logprobs": null,
          "finish_reason": "stop"
        }
      ],
      "usage": {
        "prompt_tokens": 86,
        "completion_tokens": 9,
        "total_tokens": 95,
        "prompt_tokens_details": {
          "cached_tokens": 0,
          "audio_tokens": 0
        },
        "completion_tokens_details": {
          "reasoning_tokens": 0,
          "audio_tokens": 0,
          "accepted_prediction_tokens": 0,
          "rejected_prediction_tokens": 0
        }
      },
      "service_tier": "default",
      "system_fingerprint": null
    }


* Final notes.


```python
messages = [
    {
        "role": "user",
        "content": "What's the weather like in Boston!",
    }
]
response = openai.ChatCompletion.create(
    # OpenAI Updates: As of June 2024, we are now using the GPT-3.5-Turbo model
    model="gpt-3.5-turbo",
    messages=messages,
    functions=functions,
    function_call={"name": "get_current_weather"},
)
print(response)
```

    {
      "id": "chatcmpl-BjLMqY4jQShDFUwMMSvpgCo08Bchk",
      "object": "chat.completion",
      "created": 1750146596,
      "model": "gpt-3.5-turbo-0125",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": null,
            "function_call": {
              "name": "get_current_weather",
              "arguments": "{\"location\":\"Boston\",\"unit\":\"celsius\"}"
            },
            "refusal": null,
            "annotations": []
          },
          "logprobs": null,
          "finish_reason": "stop"
        }
      ],
      "usage": {
        "prompt_tokens": 92,
        "completion_tokens": 10,
        "total_tokens": 102,
        "prompt_tokens_details": {
          "cached_tokens": 0,
          "audio_tokens": 0
        },
        "completion_tokens_details": {
          "reasoning_tokens": 0,
          "audio_tokens": 0,
          "accepted_prediction_tokens": 0,
          "rejected_prediction_tokens": 0
        }
      },
      "service_tier": "default",
      "system_fingerprint": null
    }



```python
messages.append(response["choices"][0]["message"])
```


```python
args = json.loads(response["choices"][0]["message"]['function_call']['arguments'])
observation = get_current_weather(args)
```


```python
messages.append(
        {
            "role": "function",
            "name": "get_current_weather",
            "content": observation,
        }
)
```


```python
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
)
print(response)
```

    {
      "id": "chatcmpl-BFWKVL8edWMlnpwxIT0gAvcmOrNyq",
      "object": "chat.completion",
      "created": 1743038895,
      "model": "gpt-3.5-turbo-0125",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": "The current weather in Boston is sunny and windy with a temperature of 72\u00b0F (22\u00b0C).",
            "refusal": null,
            "annotations": []
          },
          "logprobs": null,
          "finish_reason": "stop"
        }
      ],
      "usage": {
        "prompt_tokens": 84,
        "completion_tokens": 21,
        "total_tokens": 105,
        "prompt_tokens_details": {
          "cached_tokens": 0,
          "audio_tokens": 0
        },
        "completion_tokens_details": {
          "reasoning_tokens": 0,
          "audio_tokens": 0,
          "accepted_prediction_tokens": 0,
          "rejected_prediction_tokens": 0
        }
      },
      "service_tier": "default",
      "system_fingerprint": null
    }



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
