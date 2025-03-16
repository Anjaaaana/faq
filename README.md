# AI-Powered FAQ Engine

An intelligent question-answering system designed to extract precise answers from documents using advanced NLP techniques.

## 🌟 Features

- **Semantic Search**: Understands the meaning behind questions, not just keywords
- **Intelligent Answer Extraction**: Identifies the most relevant passages and extracts precise answers
- **Confidence Scoring**: Provides transparency with confidence metrics for each answer
- **Document Processing**: Automatically chunks documents while preserving question-answer pairs
- **Modern UI**: Clean, responsive interface with real-time feedback and context display

## 🛠️ Technology Stack

### Models & Components

#### Sentence Transformer
- **Model**: `all-MiniLM-L6-v2`
- **Vector Dimension**: 384
- **Purpose**: Generates document and question embeddings
- **Features**: Lightweight sentence embedding model optimized for semantic similarity

#### Question Answering Pipeline
- **Model**: `distilbert-base-cased-distilled-squad`
- **Framework**: PyTorch (`pt`)
- **Purpose**: Extracts answers from context passages
- **Training**: Fine-tuned on SQuAD (Stanford Question Answering Dataset)
- **Features**: Handles impossible answers, provides confidence scores

#### FAISS Vector Index
- **Index Type**: `IndexFlatL2`
- **Distance Metric**: L2 (Euclidean) distance
- **Purpose**: Fast approximate nearest neighbor search
- **Features**: No compression, suitable for accuracy-critical applications

### Document Processing

#### Chunking Strategy
- **Chunk Size**: 200 characters
- **Chunk Overlap**: 50 characters
- **Max Chunks to Combine**: 3
- **Max QA Pairs Per Chunk**: 5
- **Max Context Length**: 2000 characters
- **Priority**: Maintain complete QA pairs when possible

#### Intelligent Matching
- **Exact Matching**: Detects and returns exact QA pairs from document
- **Semantic Matching**: Falls back to embedding similarity search
- **Similarity Threshold**: Word overlap similarity score > 0.8 for exact matches
- **Confidence Threshold**: Responses below 0.1 are flagged as low confidence

#### Answer Processing
- **Short Answer Expansion**: Attempts to expand answers with < 5 words
- **Context Selection**: Combines top-k relevant chunks
- **QA Extraction**: Preserves full answers from detected QA pairs

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Required packages (see `requirements.txt`)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/faq-engine.git
   cd faq-engine
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare your document:
   - Place your FAQ document in the `data/` directory as `document.txt`
   - Format with "Question:" and "Answer:" prefixes for best results

4. Run the application:
   ```bash
   python app.py
   ```

5. Access the web interface:
   ```
   http://localhost:5000
   ```

## 🧰 API Endpoints

### `/ask` (POST)
Submit a question to the FAQ engine.

**Request:**
```json
{
  "question": "What is the capital of France?"
}
```

**Response:**
```json
{
  "question": "What is the capital of France?",
  "answer": "The capital of France is Paris.",
  "confidence": 0.95,
  "context_used": "Question: What is the capital of France?\nAnswer: The capital of France is Paris.",
  "low_confidence": false
}
```

### `/ask` (GET)
Submit a question via URL parameter.

**Example:**
```
GET /ask?question=What%20is%20the%20capital%20of%20France%3F
```

### `/health`
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "message": "System is running"
}
```

### `/document/info`
Get information about the loaded document.

**Response:**
```json
{
  "document_path": "data/document.txt",
  "total_chunks": 42,
  "chunk_size": 200,
  "chunk_overlap": 50,
  "embedding_model": 384,
  "total_document_size": 24680
}
```

## 🖥️ UI Features

- **Search Interface**: Clean, responsive design
- **Result Display**: Shows questions, answers, and confidence
- **Context Toggle**: Option to show source context
- **Confidence Indicator**: Visual feedback on answer reliability
- **Loading States**: Visual feedback during processing
- **Real-time Feedback**: Loading indicators during search
- **Low Confidence Warnings**: Clear indication when system is uncertain

## 💡 Performance Optimizations

- **Document Caching**: Document loaded once at initialization
- **Precomputed Embeddings**: Document embeddings computed at startup
- **Persistent Index**: FAISS index built once during initialization
- **Resource Efficiency**: Lightweight models with minimal memory footprint

## 🔒 Security Features

- **Input Validation**: Request validation to prevent injection
- **Error Handling**: Comprehensive logging and error responses
- **API Limitations**: Clean separation between frontend and backend

## 📝 Document Format Recommendations

For optimal performance, structure your document with clear question and answer patterns:

```
Question: What is machine learning?
Answer: Machine learning is a branch of artificial intelligence that focuses on building applications that learn from data and improve their accuracy over time without being explicitly programmed to do so.

Question: What are the main types of machine learning?
Answer: The main types of machine learning are supervised learning, unsupervised learning, and reinforcement learning.
```

## 🛣️ Roadmap

- [ ] Multi-document support
- [ ] Custom embedding model integration
- [ ] Document upload interface
- [ ] Answer quality metrics
- [ ] Advanced document preprocessing

## 🙏 Acknowledgements

- SentenceTransformers for the embedding models
- Hugging Face for the transformer models
- FAISS for efficient similarity search