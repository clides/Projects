from langchain_ollama import OllamaLLM

llm = OllamaLLM(model='openhermes')

print(llm("Hello"))