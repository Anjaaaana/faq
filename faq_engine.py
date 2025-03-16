from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import logging
import re
from typing import List, Tuple, Dict, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FAQEngine:
    def __init__(self, document_path, model_name='all-MiniLM-L6-v2', 
                 chunk_size=200, chunk_overlap=50, max_chunks_to_combine=3):
        """
        Initialize the FAQ Engine with improved document handling
        
        Args:
            document_path: Path to the document file
            model_name: SentenceTransformer model to use for embeddings
            chunk_size: Target size of document chunks in characters
            chunk_overlap: Overlap between chunks in characters
            max_chunks_to_combine: Maximum number of chunks to combine for context
        """
        logger.info(f"Initializing FAQEngine with document: {document_path}")
        
        # Store configuration
        self.document_path = document_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunks_to_combine = max_chunks_to_combine
        
        # Load models
        self.model = SentenceTransformer(model_name)
        logger.info(f"Loaded sentence transformer model: {model_name}")
        
        self.qa_pipeline = pipeline(
            "question-answering",
            model="distilbert-base-cased-distilled-squad",
            framework="pt"
        )
        logger.info("Loaded question-answering pipeline")
        
        # Process documents
        self.document_text = self._load_document()
        self.document_chunks, self.chunk_metadata = self._create_document_chunks()
        logger.info(f"Created {len(self.document_chunks)} document chunks")
        
        # Create and index embeddings
        self.document_embeddings = self._encode_documents()
        logger.info(f"Created document embeddings of shape {self.document_embeddings.shape}")
        
        self.index = self._build_faiss_index()
        logger.info("Built FAISS index for document embeddings")
    
    def _load_document(self) -> str:
        """Load the entire document as a single string"""
        try:
            with open(self.document_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error loading document: {str(e)}")
            raise
    
    def _create_document_chunks(self) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Create overlapping chunks from the document with metadata,
        optimized for question-answer pairs.
        
        Returns:
            Tuple containing:
                - List of document chunks
                - List of metadata dictionaries with start/end positions
        """
        document_text = self.document_text
        chunks = []
        metadata = []
        
        # First pass: try to identify question-answer sections
        qa_sections = []
        current_section = []
        lines = document_text.split('\n')
        
        for line in lines:
            # If line starts with Question: or contains it after some potential section marker
            if re.search(r'^(##?\s*)?Question:', line):
                # If we have accumulated lines in current_section, save it
                if current_section:
                    qa_sections.append('\n'.join(current_section))
                    current_section = []
                
                # Start a new section
                current_section.append(line)
            elif current_section:  # Continue adding to current section
                current_section.append(line)
        
        # Add the final section if not empty
        if current_section:
            qa_sections.append('\n'.join(current_section))
        
        # If we successfully identified QA sections, use them for chunking
        if qa_sections:
            logger.info(f"Found {len(qa_sections)} QA sections in document")
            
            # Process sections into chunks, attempting to keep related Q&A pairs together
            current_chunk = ""
            current_start = 0
            current_qa_count = 0
            max_qa_per_chunk = 5  # Maximum number of QA pairs in a chunk
            
            for section in qa_sections:
                # If adding this section would exceed size limit or max QA count
                if (len(current_chunk) + len(section) > self.chunk_size or 
                    current_qa_count >= max_qa_per_chunk) and current_chunk:
                    
                    # Save current chunk
                    chunks.append(current_chunk)
                    metadata.append({
                        "start": current_start,
                        "end": current_start + len(current_chunk),
                        "qa_count": current_qa_count
                    })
                    
                    # Start new chunk with overlap
                    # For QA format, try to maintain complete QA pairs
                    # by using the last complete QA pair as overlap
                    last_qa_pos = current_chunk.rfind("Question:")
                    if last_qa_pos > 0 and current_qa_count > 1:
                        current_chunk = current_chunk[last_qa_pos:]
                        current_start += last_qa_pos
                        current_qa_count = 1
                    else:
                        current_chunk = ""
                        current_qa_count = 0
                
                # Add this section
                if current_chunk:
                    current_chunk += "\n\n" + section
                else:
                    current_chunk = section
                
                current_qa_count += 1
            
            # Add final chunk if not empty
            if current_chunk:
                chunks.append(current_chunk)
                metadata.append({
                    "start": current_start,
                    "end": current_start + len(current_chunk),
                    "qa_count": current_qa_count
                })
        else:
            # Fall back to paragraph-based chunking
            logger.info("No QA sections identified, using paragraph-based chunking")
            
            # Break document into paragraphs
            paragraphs = [p for p in re.split(r'\n\s*\n', document_text) if p.strip()]
            
            current_chunk = ""
            current_start = 0
            
            for paragraph in paragraphs:
                # If adding this paragraph exceeds the chunk size and we already have content
                if len(current_chunk) + len(paragraph) > self.chunk_size and current_chunk:
                    # Save the current chunk
                    chunks.append(current_chunk)
                    metadata.append({
                        "start": current_start,
                        "end": current_start + len(current_chunk)
                    })
                    
                    # Start a new chunk with overlap
                    overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                    current_chunk = current_chunk[overlap_start:]
                    current_start += overlap_start
                
                # Add the paragraph (with a newline if the chunk isn't empty)
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            
            # Add the final chunk if it has content
            if current_chunk:
                chunks.append(current_chunk)
                metadata.append({
                    "start": current_start,
                    "end": current_start + len(current_chunk)
                })
        
        logger.info(f"Created {len(chunks)} chunks from document")
        return chunks, metadata
    
    def _encode_documents(self) -> np.ndarray:
        """Encode all document chunks into embeddings"""
        try:
            return self.model.encode(self.document_chunks, show_progress_bar=True)
        except Exception as e:
            logger.error(f"Error encoding documents: {str(e)}")
            raise
    
    def _build_faiss_index(self) -> faiss.IndexFlatL2:
        """Build a FAISS index for fast similarity search"""
        try:
            dimension = self.document_embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(self.document_embeddings)
            return index
        except Exception as e:
            logger.error(f"Error building FAISS index: {str(e)}")
            raise
    
    def find_relevant_chunks(self, question: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Find the most relevant document chunks for a question
        
        Args:
            question: The question to search for
            k: Number of chunks to retrieve
            
        Returns:
            List of tuples containing (chunk_text, relevance_score)
        """
        try:
            # Encode the question
            question_embedding = self.model.encode(question)
            
            # Search for similar chunks
            distances, indices = self.index.search(np.array([question_embedding]), k=k)
            
            # Convert to a list of (chunk, score) tuples
            # Note: Convert distance to similarity score (1 / (1 + distance))
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.document_chunks):  # Safety check
                    distance = distances[0][i]
                    similarity = 1 / (1 + distance)  # Convert distance to similarity
                    results.append((self.document_chunks[idx], similarity))
            
            return results
        except Exception as e:
            logger.error(f"Error finding relevant chunks: {str(e)}")
            raise
    
    def create_context(self, question: str, k: int = 5) -> str:
        """
        Create a context for the question by combining relevant chunks
        
        Args:
            question: The question to create context for
            k: Number of chunks to consider
            
        Returns:
            A string containing the combined context
        """
        relevant_chunks = self.find_relevant_chunks(question, k=k)
        
        # Determine how many chunks to use based on their size
        chunks_to_use = []
        total_length = 0
        max_context_length = 2000  # Maximum context length for the QA model
        
        # Extract full question-answer pairs if possible
        for chunk, score in relevant_chunks:
            # Look for patterns like "Question: X\nAnswer: Y" in the chunk
            if "Question:" in chunk and "Answer:" in chunk:
                qa_pairs = []
                lines = chunk.split("\n")
                current_qa = ""
                
                for i, line in enumerate(lines):
                    current_qa += line + "\n"
                    # If this is a question line and the next line is another question or end of chunk
                    if line.strip().startswith("Question:") and (i+1 >= len(lines) or lines[i+1].strip().startswith("Question:")):
                        # Add the current QA pair if it's a complete pair
                        if "Answer:" in current_qa:
                            qa_pairs.append(current_qa)
                        current_qa = ""
                
                # Add the last QA pair if it's not empty
                if current_qa and "Answer:" in current_qa:
                    qa_pairs.append(current_qa)
                
                # If we found QA pairs, use them
                if qa_pairs:
                    chunk = "\n".join(qa_pairs)
            
            if total_length + len(chunk) <= max_context_length:
                chunks_to_use.append(chunk)
                total_length += len(chunk)
                if len(chunks_to_use) >= self.max_chunks_to_combine:
                    break
            else:
                # If adding the next chunk would exceed the limit, but we have no chunks yet,
                # add a truncated version of the chunk
                if not chunks_to_use:
                    chunks_to_use.append(chunk[:max_context_length])
                break
        
        # Combine chunks into a single context
        context = "\n\n".join(chunks_to_use)
        
        # For debugging
        logger.info(f"Context created: {context[:200]}...")
        
        return context
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a question using the document chunks
        
        Args:
            question: The question to answer
            
        Returns:
            Dictionary with answer, confidence, and source information
        """
        try:
            logger.info(f"Answering question: {question}")
            
            # Find relevant context
            context = self.create_context(question)
            context_length = len(context)
            logger.info(f"Found relevant context of length {context_length}")
            
            # Try to find an exact match in the document first
            exact_answer = self._find_exact_answer(question, context)
            if exact_answer:
                logger.info(f"Found exact answer: {exact_answer}")
                return {
                    'answer': exact_answer,
                    'confidence': 0.95,  # High confidence for exact matches
                    'context_used': context[:100] + "..." if len(context) > 100 else context,
                    'low_confidence': False
                }
            
            # Use the QA pipeline to extract the answer
            result = self.qa_pipeline(question=question, context=context, 
                                     handle_impossible_answer=True)
            
            # Extract and process the answer
            answer = result['answer']
            score = result['score']
            
            # If the answer is very short, try to expand it
            if len(answer.split()) < 5:
                expanded_answer = self._expand_short_answer(answer, context)
                if expanded_answer:
                    answer = expanded_answer
            
            # If the confidence is too low, warn about it
            low_confidence = score < 0.1
            
            # Format response
            response = {
                'answer': answer,
                'confidence': score,
                'context_used': context[:100] + "..." if len(context) > 100 else context,
                'low_confidence': low_confidence
            }
            
            logger.info(f"Generated answer with confidence {score}: {answer}")
            return response
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            raise
            
    def _find_exact_answer(self, question: str, context: str) -> Optional[str]:
        """
        Look for exact question-answer pairs in the context
        
        Args:
            question: The question to find
            context: The context to search in
            
        Returns:
            The answer if an exact match is found, None otherwise
        """
        # Normalize the question to improve matching
        norm_question = question.lower().strip().rstrip('?').strip()
        
        # Look for patterns like "Question: X\nAnswer: Y" in the context
        lines = context.split('\n')
        for i, line in enumerate(lines):
            if "Question:" in line:
                # Extract the question text
                q_text = line.split("Question:", 1)[1].strip().lower().rstrip('?').strip()
                
                # Check if it's a close match
                if (norm_question in q_text or q_text in norm_question or
                    self._similarity_score(norm_question, q_text) > 0.8):
                    
                    # Look for the answer in subsequent lines
                    for j in range(i+1, min(i+5, len(lines))):
                        if "Answer:" in lines[j]:
                            # Extract the answer text
                            answer = lines[j].split("Answer:", 1)[1].strip()
                            
                            # If the next line doesn't start with "Question:", include it too
                            k = j + 1
                            while k < len(lines) and not lines[k].strip().startswith("Question:"):
                                if lines[k].strip():  # Only add non-empty lines
                                    answer += " " + lines[k].strip()
                                k += 1
                            
                            return answer
                    
        return None
    
    def _similarity_score(self, text1: str, text2: str) -> float:
        """
        Calculate a simple similarity score between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        # Very simple similarity based on word overlap
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _expand_short_answer(self, short_answer: str, context: str) -> Optional[str]:
        """
        Try to expand a short answer by finding it in the context
        
        Args:
            short_answer: The short answer to expand
            context: The context to search in
            
        Returns:
            Expanded answer if found, None otherwise
        """
        # Normalize the short answer
        norm_short = short_answer.lower().strip()
        
        # Look for the short answer within an Answer: section
        lines = context.split('\n')
        for i, line in enumerate(lines):
            if "Answer:" in line and norm_short in line.lower():
                # Extract the full answer
                answer = line.split("Answer:", 1)[1].strip()
                
                # Include subsequent lines that are part of this answer
                k = i + 1
                while k < len(lines) and not lines[k].strip().startswith("Question:"):
                    if lines[k].strip():  # Only add non-empty lines
                        answer += " " + lines[k].strip()
                    k += 1
                
                return answer
                
        return None