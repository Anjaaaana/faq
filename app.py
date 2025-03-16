from flask import Flask, request, jsonify, render_template_string
from faq_engine import FAQEngine
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Check if document exists, otherwise create a sample
document_path = 'data/document.txt'
os.makedirs(os.path.dirname(document_path), exist_ok=True)

if not os.path.exists(document_path):
    logger.warning(f"Document not found at {document_path}. Creating a sample document.")
    with open(document_path, 'w', encoding='utf-8') as f:
        f.write("""FAQ Sample Document
        
This is a sample FAQ document. Replace it with your actual document.

Question: What is the capital of France?
Answer: The capital of France is Paris.

Question: What is machine learning?
Answer: Machine learning is a branch of artificial intelligence that focuses on building applications that learn from data and improve their accuracy over time without being explicitly programmed to do so.
""")

# Initialize the FAQ engine
try:
    faq_engine = FAQEngine(document_path)
    logger.info("FAQ Engine initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize FAQ Engine: {str(e)}")
    faq_engine = None

@app.route('/')
def home():
    # Enhanced HTML template with better styling and functionality
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Enhanced FAQ System</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 900px;
                margin: 0 auto;
                padding: 20px;
                color: #333;
                line-height: 1.6;
            }
            header {
                margin-bottom: 30px;
                border-bottom: 1px solid #eee;
                padding-bottom: 20px;
            }
            h1 {
                color: #2c3e50;
                margin-bottom: 10px;
            }
            .subtitle {
                color: #7f8c8d;
                font-size: 1.1em;
                margin-top: 0;
            }
            .search-container {
                background-color: #f9f9f9;
                padding: 25px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 30px;
            }
            form {
                display: flex;
                gap: 10px;
            }
            input[type="text"] {
                flex-grow: 1;
                padding: 12px 15px;
                font-size: 16px;
                border: 1px solid #ddd;
                border-radius: 4px;
                transition: border 0.3s;
            }
            input[type="text"]:focus {
                border-color: #3498db;
                outline: none;
            }
            button {
                padding: 12px 20px;
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
                font-weight: 500;
                transition: background-color 0.3s;
            }
            button:hover {
                background-color: #2980b9;
            }
            .result-container {
                display: none;
                margin-top: 30px;
                padding: 20px;
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            .question-display {
                font-weight: 600;
                color: #2c3e50;
                font-size: 1.2em;
                margin-bottom: 15px;
                border-bottom: 1px solid #eee;
                padding-bottom: 10px;
            }
            .answer-display {
                padding: 15px;
                background-color: #f8f9fa;
                border-left: 4px solid #3498db;
                margin-bottom: 15px;
            }
            .confidence-display {
                font-size: 0.9em;
                color: #7f8c8d;
                margin-top: 10px;
            }
            .context-display {
                margin-top: 20px;
                padding: 15px;
                background-color: #f1f1f1;
                border-radius: 4px;
                font-size: 0.9em;
                display: none;
            }
            .context-toggle {
                color: #3498db;
                cursor: pointer;
                margin-top: 10px;
                font-size: 0.9em;
                display: inline-block;
            }
            .context-toggle:hover {
                text-decoration: underline;
            }
            .loading {
                color: #666;
                font-style: italic;
                margin-top: 20px;
                display: none;
            }
            .error {
                color: #e74c3c;
                padding: 15px;
                background-color: #fadbd8;
                border-radius: 4px;
            }
            .low-confidence {
                background-color: #fff3cd;
                color: #856404;
                padding: 10px;
                border-radius: 4px;
                margin-top: 10px;
                font-size: 0.9em;
            }
            @media (max-width: 768px) {
                form {
                    flex-direction: column;
                }
                button {
                    width: 100%;
                }
            }
        </style>
    </head>
    <body>
        <header>
            <h1>Enhanced FAQ System</h1>
            <p class="subtitle">Ask questions about the document to get intelligent answers</p>
        </header>
        
        <div class="search-container">
            <form id="questionForm">
                <input type="text" id="question" placeholder="What would you like to know?" required>
                <button type="submit">Ask Question</button>
            </form>
        </div>
        
        <div id="loading" class="loading">Searching for the best answer...</div>
        
        <div id="resultContainer" class="result-container">
            <div id="questionDisplay" class="question-display"></div>
            <div id="answerDisplay" class="answer-display"></div>
            <div id="confidenceDisplay" class="confidence-display"></div>
            <div id="lowConfidenceWarning" class="low-confidence" style="display: none;">
                Note: The system is not very confident in this answer. The information might not be in the document or the question may need clarification.
            </div>
            <span id="contextToggle" class="context-toggle">Show source context</span>
            <div id="contextDisplay" class="context-display"></div>
        </div>
        
        <script>
            document.getElementById('questionForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const question = document.getElementById('question').value.trim();
                
                if (!question) return;
                
                // Show loading, hide results
                document.getElementById('loading').style.display = 'block';
                document.getElementById('resultContainer').style.display = 'none';
                
                try {
                    const response = await fetch('/ask', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ question }),
                    });
                    
                    // Hide loading
                    document.getElementById('loading').style.display = 'none';
                    
                    if (response.ok) {
                        const data = await response.json();
                        
                        // Display the results
                        document.getElementById('questionDisplay').textContent = data.question;
                        document.getElementById('answerDisplay').textContent = data.answer;
                        
                        // Format and display confidence
                        const confidencePercent = Math.round(data.confidence * 100);
                        document.getElementById('confidenceDisplay').textContent = 
                            `Confidence: ${confidencePercent}%`;
                        
                        // Show or hide low confidence warning
                        document.getElementById('lowConfidenceWarning').style.display = 
                            data.low_confidence ? 'block' : 'none';
                        
                        // Set context
                        document.getElementById('contextDisplay').textContent = data.context_used;
                        
                        // Show results container
                        document.getElementById('resultContainer').style.display = 'block';
                    } else {
                        const error = await response.json();
                        // Create and display error
                        document.getElementById('resultContainer').innerHTML = `
                            <div class="error">Error: ${error.error || 'Failed to get an answer'}</div>
                        `;
                        document.getElementById('resultContainer').style.display = 'block';
                    }
                } catch (error) {
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('resultContainer').innerHTML = `
                        <div class="error">Error: Could not connect to the server</div>
                    `;
                    document.getElementById('resultContainer').style.display = 'block';
                }
            });
            
            // Toggle context visibility
            document.getElementById('contextToggle').addEventListener('click', () => {
                const contextDisplay = document.getElementById('contextDisplay');
                const contextToggle = document.getElementById('contextToggle');
                
                if (contextDisplay.style.display === 'block') {
                    contextDisplay.style.display = 'none';
                    contextToggle.textContent = 'Show source context';
                } else {
                    contextDisplay.style.display = 'block';
                    contextToggle.textContent = 'Hide source context';
                }
            });
        </script>
    </body>
    </html>
    ''')

@app.route('/ask', methods=['POST'])
def ask():
    if faq_engine is None:
        return jsonify({'error': 'FAQ Engine is not properly initialized'}), 500
        
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'Invalid JSON data'}), 400
            
        question = data.get('question')
        if not question:
            return jsonify({'error': 'No question provided'}), 400
            
        # Get answer from the FAQ engine
        response = faq_engine.answer_question(question)
        
        # Return the answer along with metadata
        return jsonify({
            'question': question,
            'answer': response['answer'],
            'confidence': response['confidence'],
            'context_used': response['context_used'],
            'low_confidence': response['low_confidence']
        })
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/ask', methods=['GET'])
def ask_get():
    if faq_engine is None:
        return jsonify({'error': 'FAQ Engine is not properly initialized'}), 500
        
    question = request.args.get('question')
    if not question:
        return jsonify({'error': 'No question provided. Add "?question=your question" to the URL'}), 400
        
    try:
        response = faq_engine.answer_question(question)
        
        return jsonify({
            'question': question,
            'answer': response['answer'],
            'confidence': response['confidence'],
            'context_used': response['context_used'],
            'low_confidence': response['low_confidence']
        })
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    if faq_engine is None:
        return jsonify({'status': 'error', 'message': 'FAQ Engine not initialized'}), 500
    return jsonify({'status': 'ok', 'message': 'System is running'})

@app.route('/document/info')
def document_info():
    """Get information about the loaded document"""
    if faq_engine is None:
        return jsonify({'error': 'FAQ Engine not initialized'}), 500
        
    try:
        return jsonify({
            'document_path': faq_engine.document_path,
            'total_chunks': len(faq_engine.document_chunks),
            'chunk_size': faq_engine.chunk_size,
            'chunk_overlap': faq_engine.chunk_overlap,
            'embedding_model': faq_engine.model.get_sentence_embedding_dimension(),
            'total_document_size': len(faq_engine.document_text)
        })
    except Exception as e:
        logger.error(f"Error getting document info: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)