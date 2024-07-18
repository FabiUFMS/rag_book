# RAG-Books
This is a little project using LangChain and Ollama to build a RAG using books as context. 

### What is RAG?
Retrieval-Augmented Generation (RAG) is the process of optimizing the output of a large language model by referencing a reliable knowledge base outside of its training data sources before generating a response.


# Steps

1. Extract text - read and extract text from a PDF file
2. Split text in chunks
3. Vector embeddings of the chunks using Chroma and Ollama embeddings
4. Load vector database
5. Test if the vector database was loaded correctly
6. Define the retriever prompt and RAG prompt
7. Configure the model and retriever
8. Create the processing chain - which contains the context provided by the retriever, question, prompt, and model
9. Ask a question through invoking the chain - the question goes through the processing chain and provides the model's answer