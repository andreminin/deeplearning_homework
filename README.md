# Deeplearning short courses homework

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
├── ACP: Agent Communication Protocol/
├── Advanced Retrieval for AI with Chroma/
├── AI Python For Beginners/
├── Attention in Transformers  Concepts and Code in PyTorch/
├── Building AI Voice Agents for Production/
├── Building Code Agents with Hugging Face smolagents/
├── Building Multimodal Search and RAG/
├── ChatGPT Prompt Engineering for Developers/
├── Federated Fine-tuning of LLMs with Private Data/
├── Finetuning Large Language Models/
├── Functions  Tools and Agents with LangChain/
├── How Diffusion Models Work/
├── How Transformer LLMs Work/
├── Introducing Multimodal Llama 3.2/
├── Intro to Federated Learning/
├── Knowledge Graphs for RAG/
├── LangChain Chat with Your Data/
├── LangChain for LLM Application Development/
├── MCP Build Rich-Context AI Apps with Anthropic/
├── Open Source Models with Hugging Face/
├── Post-training of LLMs/
├── Prompt-Compression-and-Query-optimization/
├── Quantization Fundamentals with Hugging Face/
├── Reinforcement Fine-Tuning LLMs With GRPO/
├── Reinforcement Learning From Human Feedback/
...
... more lessons 
...
└── Vector Databases: from Embeddings to Applications/
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
