# Chat Jupyter Notebooks files using Google Gemini

This Streamlit web application allows you to upload Jupyter Notebook files (.ipynb), embed them using Google's Generative AI Embeddings, and perform question-answering based on the uploaded notebooks. The app utilizes the powerful Gemini language model and its efficient inference engine to provide accurate and relevant answers to your queries. The application follows the RAG (Retrieval-Augmented Generation) approach, which combines the strengths of retrieval systems and generative language models.

## Deployed Link

Chat with .IPYNB files app is Deployed And Available [Here](https://huggingface.co/spaces/Parthiban97/Chat_With_IPYNB_Files)

## Screenshots

![Results](![Capture](https://github.com/Parthiban-R-3997/Chat_Groq_Document_Q_A/assets/26496805/e7ccdc0a-82e5-4ef2-92b5-b03dc24d519c))


## Features

- **Notebook File Upload**: Upload multiple Jupyter Notebook files (.ipynb) to create a knowledge base.
- **Document Embedding**: Leverage Google's Generative AI Embeddings to embed the uploaded notebook documents.
- **Question Answering**: Ask questions related to the uploaded notebooks, and receive precise answers powered by the Gemini model.
- **Custom Prompt Templates**: Customize the prompt template for question-answering to suit your specific needs.
- **Model Selection**: Choose from a variety of available Gemini models, including `models/gemini-1.0-pro`, `models/gemini-1.0-pro-001`, and more.
- **Document Similarity Search**: Explore relevant document chunks that match the provided question.
- **Response Time Tracking**: Monitor the response time for each query.

## RAG (Retrieval-Augmented Generation) Approach

The application follows the RAG approach, which combines the strengths of retrieval systems and generative language models. Here's how it works:

- **Retrieval**: The application embeds the uploaded notebook documents using Google's Generative AI Embeddings and stores them in a vector store (FAISS). When a user asks a question, the relevant document chunks are retrieved from the vector store based on their similarity to the question.
- **Augmentation**: The retrieved document chunks are combined and provided as context to the Gemini language model. This augments the language model's knowledge with relevant information from the uploaded documents.
- **Generation**: The Gemini language model uses the provided context and the question to generate a precise and relevant answer. The language model's generative capabilities allow it to synthesize information from the retrieved document chunks and produce a coherent response.

The RAG approach offers several advantages:

- **Scalability**: By leveraging a retrieval system, the application can handle large knowledge bases efficiently, allowing users to query information from extensive document collections.
- **Contextual Understanding**: The language model can better understand the context of the question and provide more accurate and nuanced answers by using the relevant document chunks as context.
- **Knowledge Grounding**: The generated answers are grounded in the factual information present in the uploaded documents, ensuring the reliability and trustworthiness of the responses.

## Advantages of Gemini Model

- **High Performance**: The Gemini model is designed to deliver exceptional performance for large language models, enabling fast and efficient question answering.
- **Energy Efficiency**: The model is optimized for energy efficiency, making it suitable for deployment on various devices and environments.
- **Scalability**: Gemini's architecture allows for seamless scaling of language models, ensuring that the application can handle increasing demands and larger knowledge bases.
- **Low Latency**: The model provides low-latency inference, ensuring quick response times for user queries.




