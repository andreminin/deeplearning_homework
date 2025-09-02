# Deeplearning.AI courses homework





![Deep learning](Open_Source_Models_with_Hugging_Face/labeled_image.PNG)



This repository contains Jupyter notebooks from past trainings on [DeepLearning.AI](https://learn.deeplearning.ai), adapted to run on a **local environment**.  
Some notebooks originally relied on cloud APIs (e.g., OpenAI, Cohere, Anthropic). These versions were modified to support **local LLM providers** such as:

- [Ollama](https://ollama.ai/)  
- [Xinference](https://github.com/xorbitsai/inference)  

The goal is to keep the exercises runnable offline while preserving the learning value.

---

## 📂 Repository Structure

Each course is organized into its own folder:

```
deeplearning_homework/
│
├── ACP_Agent_Communication_Protocol/
├── Advanced_Retrieval_for_AI_with_Chroma/
├── AI_Python_For_Beginners/
├── Attention_in_Transformers_Concepts_and_Code_in_PyTorch/
├── Building_AI_Voice_Agents_for_Production/
├── Building_Code_Agents_with_Hugging_Face_smolagents/
├── Building_Multimodal_Search_and_RAG/
├── ChatGPT_Prompt_Engineering_for_Developers/
├── Federated_Fine-tuning_of_LLMs_with_Private_Data/
├── Finetuning_Large_Language_Models/
├── Functions_Tools_and_Agents_with_LangChain/
├── How_Diffusion_Models_Work/
├── How_Transformer_LLMs_Work/
├── Introducing_Multimodal_Llama_3.2/
├── Intro_to_Federated_Learning/
├── Knowledge_Graphs_for_RAG/
├── LangChain_Chat_with_Your_Data/
├── LangChain_for_LLM_Application_Development/
├── MCP_Build_Rich-Context_AI_Apps_with_Anthropic/
├── Open_Source_Models_with_Hugging_Face/
├── Post-training_of_LLMs/
├── Prompt_Compression_and_Query_optimization/
├── Quantization_Fundamentals_with_Hugging_Face/
├── Reinforcement_Fine-Tuning_LLMs_With_GRPO/
├── Reinforcement_Learning_From_Human_Feedback/
...
... more lessons 
...
└── Vector_Databases_from_Embeddings_to_Applications/
```

Each directory contains:
-  Jupyter notebooks and resources for that course (original + adapted)  
- `requirements.txt` → Dependencies for running locally  

---

## 🚀 Setup Instructions

For any course folder:

```bash
cd "Course Folder Name"

# Create a virtual environment
uv venv --seed

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start Jupyter
jupyter lab
# or
jupyter notebook
# or when running notebook on remote machine
jupyter notebook --no-browser --ip=host_or_ip_or_*

# You may need to install jupyter notebook and create kernels for virtual environments:
pip install jupyter

pip install ipykernel

python -m ipykernel install --user --name=building_transformers --display-name="your-kernel-name"
```

Example:

```bash
cd "LangChain for LLM Application Development"
uv venv --seed
pip install -r requirements.txt
jupyter lab
```

---

## 📝 Notes
- Original notebooks are included and preserved where possible.  
- Adapted notebooks show how to replace cloud API calls with **local providers** (Ollama, Xinference).  
- Local LLMs like **LLaMA 2**, **Mistral**, **Qwen**, or **Llama 3.2** can be tested with these setups.  

---

## 📄 License
This project is for **personal, educational use only**.  
All credit for the original notebooks goes to [DeepLearning.AI](https://learn.deeplearning.ai).  

---
