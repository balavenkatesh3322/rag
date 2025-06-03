FROM python:3.10-slim

# Install dependencies
RUN apt-get update && apt-get install -y curl git

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Preload model
# RUN ollama pull nomic-embed-text:latest
# RUN ollama pull llama3.2:1b
# RUN ollama pull qwen3:8b

# Copy your app
WORKDIR /app
COPY . /app

# Install Python dependencies
RUN pip install -r requirements.txt

# Expose Streamlit and Ollama ports
EXPOSE 7860 11434

# Run both Ollama and Streamlit
# Start Ollama server, pull model, and run Streamlit app
CMD ollama serve & \
    sleep 5 && \
    ollama pull nomic-embed-text && \
    ollama pull llama3.2:1b && \
    python run.py --server.port=7860 --server.address=0.0.0.0
