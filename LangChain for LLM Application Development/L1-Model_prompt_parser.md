# LangChain: Models, Prompts and Output Parsers


## Outline

 * Direct API calls to OpenAI
 * API calls through LangChain:
   * Prompts
   * Models
   * Output parsers

## Get your [OpenAI API Key](https://platform.openai.com/account/api-keys)


```python
#!pip install python-dotenv
#!pip install openai
```


```python
import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']
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
print(llm_model)
```

    gpt-3.5-turbo


## Chat API : OpenAI

Let's start with a direct API calls to OpenAI.


```python
def get_completion(prompt, model=llm_model):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, 
    )
    return response.choices[0].message["content"]

```


```python
get_completion("What is square root from 5 rounded 2 numbers after dot?")
```




    'The square root of 5 rounded to 2 decimal places is approximately 2.24.'




```python
customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse,\
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""
```


```python
style = """American English \
in a calm and respectful tone
"""
```


```python
prompt = f"""Translate the text \
that is delimited by triple backticks 
into a style that is {style}.
text: ```{customer_email}```
"""

print(prompt)
```

    Translate the text that is delimited by triple backticks 
    into a style that is American English in a calm and respectful tone
    .
    text: ```
    Arrr, I be fuming that me blender lid flew off and splattered me kitchen walls with smoothie! And to make matters worse,the warranty don't cover the cost of cleaning up me kitchen. I need yer help right now, matey!
    ```
    



```python
response = get_completion(prompt)
```


```python
response
```




    "Ah, I'm really frustrated that my blender lid flew off and splattered my kitchen walls with smoothie! And to make matters worse, the warranty doesn't cover the cost of cleaning up my kitchen. I could really use your help right now, friend."



## Chat API : LangChain

Let's try how we can do the same using LangChain.


```python
#!pip install --upgrade langchain
```

### Model


```python
from langchain.chat_models import ChatOpenAI
```


```python
# To control the randomness and creativity of the generated
# text by an LLM, use temperature = 0.0
chat = ChatOpenAI(temperature=0.0, model=llm_model)
chat
```




    ChatOpenAI(verbose=False, callbacks=None, callback_manager=None, client=<class 'openai.api_resources.chat_completion.ChatCompletion'>, model_name='gpt-3.5-turbo', temperature=0.0, model_kwargs={}, openai_api_key=None, openai_api_base=None, openai_organization=None, request_timeout=None, max_retries=6, streaming=False, n=1, max_tokens=None)



### Prompt template


```python
template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{text}```
"""
```


```python
from langchain.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_template(template_string)

```


```python
prompt_template.messages[0].prompt
```




    PromptTemplate(input_variables=['style', 'text'], output_parser=None, partial_variables={}, template='Translate the text that is delimited by triple backticks into a style that is {style}. text: ```{text}```\n', template_format='f-string', validate_template=True)




```python
prompt_template.messages[0].prompt.input_variables
```




    ['style', 'text']




```python
customer_style = """American English \
in a calm and respectful tone
"""
```


```python
customer_style = """Русский \
в спокойном и уважительном тоне
"""
```


```python
customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse, \
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""
```


```python
customer_messages = prompt_template.format_messages(
                    style=customer_style,
                    text=customer_email)
```


```python
print(type(customer_messages))
print(type(customer_messages[0]))
```

    <class 'list'>
    <class 'langchain.schema.HumanMessage'>



```python
print(customer_messages[0])
```

    content="Translate the text that is delimited by triple backticks into a style that is Русский в спокойном и уважительном тоне\n. text: ```\nArrr, I be fuming that me blender lid flew off and splattered me kitchen walls with smoothie! And to make matters worse, the warranty don't cover the cost of cleaning up me kitchen. I need yer help right now, matey!\n```\n" additional_kwargs={} example=False



```python
# Call the LLM to translate to the style of the customer message
customer_response = chat(customer_messages)
```


```python
print(customer_response.content)
```

    Да уж, я весь в ярости, что крышка моего блендера отлетела и покрасила стены моей кухни смузи! И чтобы дела стали еще хуже, гарантия не покрывает расходы на уборку моей кухни. Мне нужна твоя помощь прямо сейчас, приятель!



```python
service_reply = """Hey there customer, \
the warranty does not cover \
cleaning expenses for your kitchen \
because it's your fault that \
you misused your blender \
by forgetting to put the lid on before \
starting the blender. \
Tough luck! See ya!
"""
```


```python

```


```python
service_style_pirate = """\
a polite tone \
that speaks in English Pirate\
"""
```


```python
service_style_pirate = """\
Переведи в тоне беспредельщика в русском бандитском разговорном сленге по понятиям\
"""
```


```python

```


```python
service_messages = prompt_template.format_messages(
    style=service_style_pirate,
    text=service_reply)

print(service_messages[0].content)
```

    Translate the text that is delimited by triple backticks into a style that is Переведи в тоне беспредельщика в русском бандитском разговорном сленге по понятиям. text: ```Hey there customer, the warranty does not cover cleaning expenses for your kitchen because it's your fault that you misused your blender by forgetting to put the lid on before starting the blender. Tough luck! See ya!
    ```
    



```python

```


```python
service_response = chat(service_messages)
print(service_response.content)
```

    Слышь, клиент, гарантия не распространяется на расходы на уборку для твоей кухни, потому что ты сам виноват, что неправильно использовал блендер, забыв поставить крышку перед включением. Не повезло, братан! До встречи!



```python

```

## Output Parsers

Let's start with defining how we would like the LLM output to look like:


```python
{
  "gift": False,
  "delivery_days": 5,
  "price_value": "pretty affordable!"
}
```




    {'gift': False, 'delivery_days': 5, 'price_value': 'pretty affordable!'}




```python
customer_review = """\
This leaf blower is pretty amazing.  It has four settings:\
candle blower, gentle breeze, windy city, and tornado. \
It arrived in two days, just in time for my wife's \
anniversary present. \
I think my wife liked it so much she was speechless. \
So far I've been the only one using it, and I've been \
using it every other morning to clear the leaves on our lawn. \
It's slightly more expensive than the other leaf blowers \
out there, but I think it's worth it for the extra features.
"""

review_template = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product \
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

Format the output as JSON with the following keys:
gift
delivery_days
price_value

text: {text}
"""
```


```python
from langchain.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_template(review_template)
print(prompt_template)
```

    input_variables=['text'] output_parser=None partial_variables={} messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['text'], output_parser=None, partial_variables={}, template='For the following text, extract the following information:\n\ngift: Was the item purchased as a gift for someone else? Answer True if yes, False if not or unknown.\n\ndelivery_days: How many days did it take for the product to arrive? If this information is not found, output -1.\n\nprice_value: Extract any sentences about the value or price,and output them as a comma separated Python list.\n\nFormat the output as JSON with the following keys:\ngift\ndelivery_days\nprice_value\n\ntext: {text}\n', template_format='f-string', validate_template=True), additional_kwargs={})]



```python
messages = prompt_template.format_messages(text=customer_review)
chat = ChatOpenAI(temperature=0.0, model=llm_model)
response = chat(messages)
print(response.content)
```

    {
      "gift": true,
      "delivery_days": 2,
      "price_value": "It's slightly more expensive than the other leaf blowers out there"
    }



```python
type(response.content)
```




    str




```python
# You will get an error by running this line of code 
# because'gift' is not a dictionary
# 'gift' is a string
response.content.get('gift')
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    Cell In[52], line 4
          1 # You will get an error by running this line of code 
          2 # because'gift' is not a dictionary
          3 # 'gift' is a string
    ----> 4 response.content.get('gift')


    AttributeError: 'str' object has no attribute 'get'


### Parse the LLM output string into a Python dictionary


```python
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
```


```python
gift_schema = ResponseSchema(name="gift",
                             description="Was the item purchased\
                             as a gift for someone else? \
                             Answer True if yes,\
                             False if not or unknown.")
delivery_days_schema = ResponseSchema(name="delivery_days",
                                      description="How many days\
                                      did it take for the product\
                                      to arrive? If this \
                                      information is not found,\
                                      output -1.")
price_value_schema = ResponseSchema(name="price_value",
                                    description="Extract any\
                                    sentences about the value or \
                                    price, and output them as a \
                                    comma separated Python list.")

response_schemas = [gift_schema, 
                    delivery_days_schema,
                    price_value_schema]
```


```python
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
```


```python
format_instructions = output_parser.get_format_instructions()
```


```python
print(format_instructions)
```

    The output should be a markdown code snippet formatted in the following schema, including the leading and trailing "\`\`\`json" and "\`\`\`":
    
    ```json
    {
    	"gift": string  // Was the item purchased                             as a gift for someone else?                              Answer True if yes,                             False if not or unknown.
    	"delivery_days": string  // How many days                                      did it take for the product                                      to arrive? If this                                       information is not found,                                      output -1.
    	"price_value": string  // Extract any                                    sentences about the value or                                     price, and output them as a                                     comma separated Python list.
    }
    ```



```python
review_template_2 = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product\
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

text: {text}

{format_instructions}
"""

prompt = ChatPromptTemplate.from_template(template=review_template_2)

messages = prompt.format_messages(text=customer_review, 
                                format_instructions=format_instructions)
```


```python
print(messages[0].content)
```

    For the following text, extract the following information:
    
    gift: Was the item purchased as a gift for someone else? Answer True if yes, False if not or unknown.
    
    delivery_days: How many days did it take for the productto arrive? If this information is not found, output -1.
    
    price_value: Extract any sentences about the value or price,and output them as a comma separated Python list.
    
    text: This leaf blower is pretty amazing.  It has four settings:candle blower, gentle breeze, windy city, and tornado. It arrived in two days, just in time for my wife's anniversary present. I think my wife liked it so much she was speechless. So far I've been the only one using it, and I've been using it every other morning to clear the leaves on our lawn. It's slightly more expensive than the other leaf blowers out there, but I think it's worth it for the extra features.
    
    
    The output should be a markdown code snippet formatted in the following schema, including the leading and trailing "\`\`\`json" and "\`\`\`":
    
    ```json
    {
    	"gift": string  // Was the item purchased                             as a gift for someone else?                              Answer True if yes,                             False if not or unknown.
    	"delivery_days": string  // How many days                                      did it take for the product                                      to arrive? If this                                       information is not found,                                      output -1.
    	"price_value": string  // Extract any                                    sentences about the value or                                     price, and output them as a                                     comma separated Python list.
    }
    ```
    



```python
response = chat(messages)
```


```python
print(response.content)
```

    ```json
    {
    	"gift": true,
    	"delivery_days": 2,
    	"price_value": ["It's slightly more expensive than the other leaf blowers out there, but I think it's worth it for the extra features."]
    }
    ```



```python
output_dict = output_parser.parse(response.content)
```


```python
output_dict
```




    {'gift': True,
     'delivery_days': 2,
     'price_value': ["It's slightly more expensive than the other leaf blowers out there, but I think it's worth it for the extra features."]}




```python
type(output_dict)
```




    dict




```python
output_dict.get('delivery_days')
```




    2



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
