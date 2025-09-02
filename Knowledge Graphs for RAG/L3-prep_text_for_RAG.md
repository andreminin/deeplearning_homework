# Lesson 3: Preparing Text Data for RAG

<p style="background-color:#fd4a6180; padding:15px; margin-left:20px"> <b>Note:</b> This notebook takes about 30 seconds to be ready to use. Please wait until the "Kernel starting, please wait..." message clears from the top of the notebook before running any cells. You may start the video while you wait.</p>


### Import packages and set up Neo4j


```python
from dotenv import load_dotenv
import os

from langchain_community.graphs import Neo4jGraph

# Warning control
import warnings
warnings.filterwarnings("ignore")
```


```python
# Load from environment
load_dotenv('.env', override=True)
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Note the code below is unique to this course environment, and not a 
# standard part of Neo4j's integration with OpenAI. Remove if running 
# in your own environment.
OPENAI_ENDPOINT = os.getenv('OPENAI_BASE_URL') + '/embeddings'
```


```python
# Connect to the knowledge graph instance using LangChain
kg = Neo4jGraph(
    url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE
)
```

### Create a vector index 


```python
kg.query("""
  CREATE VECTOR INDEX movie_tagline_embeddings IF NOT EXISTS
  FOR (m:Movie) ON (m.taglineEmbedding) 
  OPTIONS { indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
  }}"""
)

```




    []




```python
kg.query("""
  SHOW VECTOR INDEXES
  """
)
```




    [{'id': 3,
      'name': 'movie_tagline_embeddings',
      'state': 'ONLINE',
      'populationPercent': 100.0,
      'type': 'VECTOR',
      'entityType': 'NODE',
      'labelsOrTypes': ['Movie'],
      'properties': ['taglineEmbedding'],
      'indexProvider': 'vector-1.0',
      'owningConstraint': None,
      'lastRead': None,
      'readCount': None}]




```python

```

### Populate the vector index
- Calculate vector representation for each movie tagline using OpenAI
- Add vector to the `Movie` node as `taglineEmbedding` property


```python
kg.query("""
    MATCH (movie:Movie) WHERE movie.tagline IS NOT NULL
    WITH movie, genai.vector.encode(
        movie.tagline, 
        "OpenAI", 
        {
          token: $openAiApiKey,
          endpoint: $openAiEndpoint
        }) AS vector
    CALL db.create.setNodeVectorProperty(movie, "taglineEmbedding", vector)
    """, 
    params={"openAiApiKey":OPENAI_API_KEY, "openAiEndpoint": OPENAI_ENDPOINT} )
```




    []




```python
result = kg.query("""
    MATCH (m:Movie) 
    WHERE m.tagline IS NOT NULL
    RETURN m.tagline, m.taglineEmbedding
    LIMIT 1
    """
)
```


```python
result[0]['m.tagline']
```




    'Welcome to the Real World'




```python
result[0]['m.taglineEmbedding'][:10]
```




    [0.017445066943764687,
     -0.005481892731040716,
     -0.002013522433117032,
     -0.025571243837475777,
     -0.014404304325580597,
     0.016737302765250206,
     -0.017078077420592308,
     0.000485358847072348,
     -0.025217361748218536,
     -0.029516370967030525]




```python
len(result[0]['m.taglineEmbedding'])
```




    1536




```python

```

### Similarity search
- Calculate embedding for question
- Identify matching movies based on similarity of question and `taglineEmbedding` vectors


```python
question = "What movies are about love?"
```


```python
kg.query("""
    WITH genai.vector.encode(
        $question, 
        "OpenAI", 
        {
          token: $openAiApiKey,
          endpoint: $openAiEndpoint
        }) AS question_embedding
    CALL db.index.vector.queryNodes(
        'movie_tagline_embeddings', 
        $top_k, 
        question_embedding
        ) YIELD node AS movie, score
    RETURN movie.title, movie.tagline, score
    """, 
    params={"openAiApiKey":OPENAI_API_KEY,
            "openAiEndpoint": OPENAI_ENDPOINT,
            "question": question,
            "top_k": 5
            })
```




    [{'movie.title': 'Joe Versus the Volcano',
      'movie.tagline': 'A story of love, lava and burning desire.',
      'score': 0.9062913656234741},
     {'movie.title': 'As Good as It Gets',
      'movie.tagline': 'A comedy from the heart that goes for the throat.',
      'score': 0.9022631645202637},
     {'movie.title': 'Snow Falling on Cedars',
      'movie.tagline': 'First loves last. Forever.',
      'score': 0.9013131856918335},
     {'movie.title': 'Sleepless in Seattle',
      'movie.tagline': 'What if someone you never met, someone you never saw, someone you never knew was the only someone for you?',
      'score': 0.8945093154907227},
     {'movie.title': 'When Harry Met Sally',
      'movie.tagline': 'Can two friends sleep together and still love each other in the morning?',
      'score': 0.894220232963562}]



### Try for yourself: ask you own question!
- Change the question below and run the graph query to find different movies


```python
question = "What movies are about adventure?"
```


```python
kg.query("""
    WITH genai.vector.encode(
        $question, 
        "OpenAI", 
        {
          token: $openAiApiKey,
          endpoint: $openAiEndpoint
        }) AS question_embedding
    CALL db.index.vector.queryNodes(
        'movie_tagline_embeddings', 
        $top_k, 
        question_embedding
        ) YIELD node AS movie, score
    RETURN movie.title, movie.tagline, score
    """, 
    params={"openAiApiKey":OPENAI_API_KEY,
            "openAiEndpoint": OPENAI_ENDPOINT,
            "question": question,
            "top_k": 5
            })
```




    [{'movie.title': 'RescueDawn',
      'movie.tagline': "Based on the extraordinary true story of one man's fight for freedom",
      'score': 0.8998090028762817},
     {'movie.title': 'Cast Away',
      'movie.tagline': 'At the edge of the world, his journey begins.',
      'score': 0.8982737064361572},
     {'movie.title': 'Ninja Assassin',
      'movie.tagline': 'Prepare to enter a secret world of assassins',
      'score': 0.8880558013916016},
     {'movie.title': 'Joe Versus the Volcano',
      'movie.tagline': 'A story of love, lava and burning desire.',
      'score': 0.8870121240615845},
     {'movie.title': 'As Good as It Gets',
      'movie.tagline': 'A comedy from the heart that goes for the throat.',
      'score': 0.8856385350227356}]




```python

```
