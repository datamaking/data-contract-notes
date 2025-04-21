from typing import List, Dict, Tuple, Optional, Union, Callable
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, explode, array, split, size
from pyspark.sql.types import ArrayType, StringType, StructType, StructField, IntegerType
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class ChunkingStrategy:
    """Base class for all chunking strategies."""
    
    def __init__(self, spark: Optional[SparkSession] = None):
        """
        Initialize the chunking strategy.
        
        Args:
            spark: SparkSession instance. If None, will create a new one.
        """
        if spark is None:
            self.spark = SparkSession.builder \
                .appName("TextChunking") \
                .master("local[*]") \
                .getOrCreate()
        else:
            self.spark = spark
            
    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk the input text according to the strategy.
        
        Args:
            text: Input text to be chunked.
            
        Returns:
            List of text chunks.
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def chunk_dataframe(self, df, text_col: str, output_col: str = "chunks"):
        """
        Apply chunking to a DataFrame column.
        
        Args:
            df: Input DataFrame
            text_col: Column name containing text to chunk
            output_col: Column name for output chunks
            
        Returns:
            DataFrame with chunks added as a new column
        """
        chunk_udf = udf(self.chunk_text, ArrayType(StringType()))
        return df.withColumn(output_col, chunk_udf(col(text_col)))


class FixedLengthChunking(ChunkingStrategy):
    """Chunk text into fixed-length segments."""
    
    def __init__(self, chunk_size: int = 100, overlap: int = 0, **kwargs):
        """
        Initialize fixed-length chunking.
        
        Args:
            chunk_size: Number of tokens in each chunk
            overlap: Number of tokens to overlap between chunks
            **kwargs: Additional arguments for the base class
        """
        super().__init__(**kwargs)
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text into fixed-length segments.
        
        Args:
            text: Input text to be chunked
            
        Returns:
            List of text chunks
        """
        if not text or not isinstance(text, str):
            return []
            
        tokens = word_tokenize(text)
        chunks = []
        
        if self.chunk_size <= 0:
            return [text]
            
        i = 0
        while i < len(tokens):
            chunk_end = min(i + self.chunk_size, len(tokens))
            chunks.append(" ".join(tokens[i:chunk_end]))
            i += self.chunk_size - self.overlap
            
        return chunks


class SentenceBasedChunking(ChunkingStrategy):
    """Chunk text based on sentences."""
    
    def __init__(self, max_sentences: int = 5, **kwargs):
        """
        Initialize sentence-based chunking.
        
        Args:
            max_sentences: Maximum number of sentences per chunk
            **kwargs: Additional arguments for the base class
        """
        super().__init__(**kwargs)
        self.max_sentences = max_sentences
        
    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text based on sentences.
        
        Args:
            text: Input text to be chunked
            
        Returns:
            List of text chunks
        """
        if not text or not isinstance(text, str):
            return []
            
        sentences = sent_tokenize(text)
        chunks = []
        
        for i in range(0, len(sentences), self.max_sentences):
            chunk = " ".join(sentences[i:i + self.max_sentences])
            chunks.append(chunk)
            
        return chunks


class ParagraphBasedChunking(ChunkingStrategy):
    """Chunk text based on paragraphs."""
    
    def __init__(self, max_paragraphs: int = 1, **kwargs):
        """
        Initialize paragraph-based chunking.
        
        Args:
            max_paragraphs: Maximum number of paragraphs per chunk
            **kwargs: Additional arguments for the base class
        """
        super().__init__(**kwargs)
        self.max_paragraphs = max_paragraphs
        
    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text based on paragraphs.
        
        Args:
            text: Input text to be chunked
            
        Returns:
            List of text chunks
        """
        if not text or not isinstance(text, str):
            return []
            
        # Split text by double newlines or other paragraph separators
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        chunks = []
        
        for i in range(0, len(paragraphs), self.max_paragraphs):
            chunk = "\n\n".join(paragraphs[i:i + self.max_paragraphs])
            chunks.append(chunk)
            
        return chunks


class SlidingWindowChunking(ChunkingStrategy):
    """Chunk text using a sliding window approach."""
    
    def __init__(self, window_size: int = 100, step_size: int = 50, **kwargs):
        """
        Initialize sliding window chunking.
        
        Args:
            window_size: Size of the sliding window in tokens
            step_size: Number of tokens to slide the window
            **kwargs: Additional arguments for the base class
        """
        super().__init__(**kwargs)
        self.window_size = window_size
        self.step_size = step_size
        
    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text using a sliding window approach.
        
        Args:
            text: Input text to be chunked
            
        Returns:
            List of text chunks
        """
        if not text or not isinstance(text, str):
            return []
            
        tokens = word_tokenize(text)
        chunks = []
        
        for i in range(0, len(tokens) - self.window_size + 1, self.step_size):
            chunk = " ".join(tokens[i:i + self.window_size])
            chunks.append(chunk)
            
        # Handle the last chunk if it doesn't align perfectly
        if len(tokens) > 0 and (len(tokens) - self.window_size) % self.step_size != 0:
            last_chunk = " ".join(tokens[-self.window_size:])
            if last_chunk not in chunks:
                chunks.append(last_chunk)
                
        return chunks


class SemanticChunking(ChunkingStrategy):
    """Chunk text based on semantic similarity."""
    
    def __init__(self, similarity_threshold: float = 0.5, max_chunk_size: int = 200, **kwargs):
        """
        Initialize semantic chunking.
        
        Args:
            similarity_threshold: Threshold for semantic similarity
            max_chunk_size: Maximum size of a chunk in tokens
            **kwargs: Additional arguments for the base class
        """
        super().__init__(**kwargs)
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
        self.vectorizer = TfidfVectorizer()
        
    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text based on semantic similarity.
        
        Args:
            text: Input text to be chunked
            
        Returns:
            List of text chunks
        """
        if not text or not isinstance(text, str):
            return []
            
        sentences = sent_tokenize(text)
        if len(sentences) <= 1:
            return sentences
            
        # Vectorize sentences
        try:
            sentence_vectors = self.vectorizer.fit_transform(sentences)
            similarity_matrix = cosine_similarity(sentence_vectors)
        except:
            # Fallback to sentence-based chunking if vectorization fails
            return SentenceBasedChunking().chunk_text(text)
        
        chunks = []
        current_chunk = [sentences[0]]
        current_chunk_size = len(word_tokenize(sentences[0]))
        
        for i in range(1, len(sentences)):
            # Calculate average similarity with current chunk
            avg_similarity = np.mean([similarity_matrix[i][j] for j in range(i-len(current_chunk), i)])
            current_sent_size = len(word_tokenize(sentences[i]))
            
            # Check if sentence is semantically similar and chunk isn't too large
            if avg_similarity >= self.similarity_threshold and current_chunk_size + current_sent_size <= self.max_chunk_size:
                current_chunk.append(sentences[i])
                current_chunk_size += current_sent_size
            else:
                # Save current chunk and start a new one
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]
                current_chunk_size = current_sent_size
                
        # Add the last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks


class RecursiveChunking(ChunkingStrategy):
    """Recursively chunk text based on document structure."""
    
    def __init__(self, max_chunk_size: int = 1000, **kwargs):
        """
        Initialize recursive chunking.
        
        Args:
            max_chunk_size: Maximum size of a chunk in tokens
            **kwargs: Additional arguments for the base class
        """
        super().__init__(**kwargs)
        self.max_chunk_size = max_chunk_size
        
    def chunk_text(self, text: str) -> List[str]:
        """
        Recursively chunk text based on document structure.
        
        Args:
            text: Input text to be chunked
            
        Returns:
            List of text chunks
        """
        if not text or not isinstance(text, str):
            return []
            
        # Initial split by sections (e.g., headers)
        sections = self._split_by_headers(text)
        chunks = []
        
        for section in sections:
            section_tokens = word_tokenize(section)
            
            if len(section_tokens) <= self.max_chunk_size:
                chunks.append(section)
            else:
                # Recursively split large sections
                paragraphs = re.split(r'\n\s*\n', section)
                for paragraph in paragraphs:
                    if len(word_tokenize(paragraph)) <= self.max_chunk_size:
                        chunks.append(paragraph)
                    else:
                        # Further split large paragraphs into sentences
                        sentences = sent_tokenize(paragraph)
                        current_chunk = []
                        current_size = 0
                        
                        for sentence in sentences:
                            sentence_size = len(word_tokenize(sentence))
                            if current_size + sentence_size <= self.max_chunk_size:
                                current_chunk.append(sentence)
                                current_size += sentence_size
                            else:
                                if current_chunk:
                                    chunks.append(" ".join(current_chunk))
                                
                                # If a single sentence is too large, split it by fixed length
                                if sentence_size > self.max_chunk_size:
                                    sentence_tokens = word_tokenize(sentence)
                                    for i in range(0, len(sentence_tokens), self.max_chunk_size):
                                        chunks.append(" ".join(sentence_tokens[i:i + self.max_chunk_size]))
                                else:
                                    current_chunk = [sentence]
                                    current_size = sentence_size
                        
                        if current_chunk:
                            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _split_by_headers(self, text: str) -> List[str]:
        """Split text by header patterns."""
        # Simple header pattern matching (can be extended for more complex documents)
        header_pattern = r'(?:\n|^)#+\s+.+?(?=\n|$)'
        splits = re.split(header_pattern, text)
        headers = re.findall(header_pattern, text)
        
        # Recombine headers with their content
        sections = []
        for i in range(len(headers)):
            if i < len(splits):
                sections.append(headers[i] + splits[i+1])
        
        # Add first section if it exists
        if splits[0].strip():
            sections.insert(0, splits[0])
            
        return sections


class ContextEnrichedChunking(ChunkingStrategy):
    """Chunk text while preserving context from surrounding text."""
    
    def __init__(self, base_strategy: ChunkingStrategy, context_size: int = 50, **kwargs):
        """
        Initialize context-enriched chunking.
        
        Args:
            base_strategy: Base chunking strategy to use
            context_size: Number of tokens to include as context
            **kwargs: Additional arguments for the base class
        """
        super().__init__(**kwargs)
        self.base_strategy = base_strategy
        self.context_size = context_size
        
    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text while preserving context.
        
        Args:
            text: Input text to be chunked
            
        Returns:
            List of text chunks with context
        """
        if not text or not isinstance(text, str):
            return []
            
        # Get base chunks
        base_chunks = self.base_strategy.chunk_text(text)
        if not base_chunks:
            return []
            
        tokens = word_tokenize(text)
        enriched_chunks = []
        
        for chunk in base_chunks:
            # Find position of chunk in original text
            chunk_tokens = word_tokenize(chunk)
            chunk_start = self._find_subsequence(tokens, chunk_tokens)
            
            if chunk_start is not None:
                # Add context before and after
                context_start = max(0, chunk_start - self.context_size)
                context_end = min(len(tokens), chunk_start + len(chunk_tokens) + self.context_size)
                
                # Create enriched chunk
                prefix = " ".join(tokens[context_start:chunk_start])
                suffix = " ".join(tokens[chunk_start + len(chunk_tokens):context_end])
                
                enriched_chunk = f"[CONTEXT_PREFIX] {prefix} [CHUNK_START] {chunk} [CHUNK_END] {suffix} [CONTEXT_SUFFIX]"
                enriched_chunks.append(enriched_chunk)
            else:
                # If chunk position can't be determined, use original chunk
                enriched_chunks.append(chunk)
                
        return enriched_chunks
    
    def _find_subsequence(self, seq, subseq):
        """Find the starting position of a subsequence in a sequence."""
        n, m = len(seq), len(subseq)
        for i in range(n - m + 1):
            if seq[i:i+m] == subseq:
                return i
        return None


class ModalitySpecificChunking(ChunkingStrategy):
    """Chunk text based on specific modalities or content types."""
    
    def __init__(self, **kwargs):
        """
        Initialize modality-specific chunking.
        
        Args:
            **kwargs: Additional arguments for the base class
        """
        super().__init__(**kwargs)
        self.modality_patterns = {
            'code': r'```[\s\S]*?```|`[^`\n]+`',
            'table': r'\|.*\|[\s]*\n\|[\s]*[-:]+[-|\s:]*\n(\|.*\|[\s]*\n)+',
            'list': r'(?:\n|^)(?:[\*\-\+]|\d+\.)(?:\s+).*(?:\n(?:[\*\-\+]|\d+\.)(?:\s+).*)*',
            'equation': r'\$\$[\s\S]*?\$\$|\$[^$\n]+\$',
        }
        
    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text based on specific modalities.
        
        Args:
            text: Input text to be chunked
            
        Returns:
            List of text chunks with modality information
        """
        if not text or not isinstance(text, str):
            return []
            
        # Extract special modalities
        special_sections = []
        for modality, pattern in self.modality_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                special_sections.append({
                    'start': match.start(),
                    'end': match.end(),
                    'text': match.group(),
                    'modality': modality
                })
        
        # Sort by position
        special_sections.sort(key=lambda x: x['start'])
        
        # Extract regular text sections
        chunks = []
        last_end = 0
        
        for section in special_sections:
            # Add text before special section
            if section['start'] > last_end:
                regular_text = text[last_end:section['start']].strip()
                if regular_text:
                    chunks.append(f"[MODALITY:text] {regular_text}")
            
            # Add special section
            chunks.append(f"[MODALITY:{section['modality']}] {section['text']}")
            last_end = section['end']
        
        # Add remaining text
        if last_end < len(text):
            regular_text = text[last_end:].strip()
            if regular_text:
                chunks.append(f"[MODALITY:text] {regular_text}")
                
        return chunks


class AgenticChunking(ChunkingStrategy):
    """Chunk text based on agent-specific criteria."""
    
    def __init__(self, agent_rules: Dict[str, Callable[[str], bool]], **kwargs):
        """
        Initialize agentic chunking.
        
        Args:
            agent_rules: Dictionary mapping agent names to functions that determine relevance
            **kwargs: Additional arguments for the base class
        """
        super().__init__(**kwargs)
        self.agent_rules = agent_rules
        
    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text based on agent-specific criteria.
        
        Args:
            text: Input text to be chunked
            
        Returns:
            List of text chunks with agent tags
        """
        if not text or not isinstance(text, str):
            return []
            
        # Split into sentences for evaluation
        sentences = sent_tokenize(text)
        agent_chunks = {agent: [] for agent in self.agent_rules.keys()}
        
        # Assign sentences to agents
        for sentence in sentences:
            for agent, rule in self.agent_rules.items():
                if rule(sentence):
                    agent_chunks[agent].append(sentence)
        
        # Create final chunks
        chunks = []
        for agent, sentences in agent_chunks.items():
            if sentences:
                chunks.append(f"[AGENT:{agent}] {' '.join(sentences)}")
                
        return chunks


class SubdocumentChunking(ChunkingStrategy):
    """Chunk text based on subdocument boundaries."""
    
    def __init__(self, subdoc_patterns: Dict[str, str], **kwargs):
        """
        Initialize subdocument chunking.
        
        Args:
            subdoc_patterns: Dictionary mapping subdocument types to regex patterns
            **kwargs: Additional arguments for the base class
        """
        super().__init__(**kwargs)
        self.subdoc_patterns = subdoc_patterns
        
    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text based on subdocument boundaries.
        
        Args:
            text: Input text to be chunked
            
        Returns:
            List of text chunks with subdocument type
        """
        if not text or not isinstance(text, str):
            return []
            
        chunks = []
        remaining_text = text
        
        # Extract subdocuments
        for subdoc_type, pattern in self.subdoc_patterns.items():
            matches = list(re.finditer(pattern, remaining_text))
            
            for match in reversed(matches):  # Process in reverse to maintain indices
                subdoc = match.group()
                chunks.append(f"[SUBDOC:{subdoc_type}] {subdoc}")
                
                # Remove matched text from remaining text
                remaining_text = remaining_text[:match.start()] + remaining_text[match.end():]
        
        # Add remaining text as a general chunk
        if remaining_text.strip():
            chunks.append(f"[SUBDOC:general] {remaining_text.strip()}")
            
        # Sort chunks based on their original position in the text
        # This is a simplification - actual implementation would need to track positions
        return chunks


class HybridChunking(ChunkingStrategy):
    """Combine multiple chunking strategies."""
    
    def __init__(self, strategies: List[Tuple[ChunkingStrategy, float]], **kwargs):
        """
        Initialize hybrid chunking.
        
        Args:
            strategies: List of (strategy, weight) tuples
            **kwargs: Additional arguments for the base class
        """
        super().__init__(**kwargs)
        self.strategies = strategies
        
    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text using multiple strategies.
        
        Args:
            text: Input text to be chunked
            
        Returns:
            List of text chunks
        """
        if not text or not isinstance(text, str):
            return []
            
        all_chunks = []
        
        # Apply each strategy
        for strategy, weight in self.strategies:
            chunks = strategy.chunk_text(text)
            for chunk in chunks:
                all_chunks.append({
                    'text': chunk,
                    'weight': weight,
                    'strategy': strategy.__class__.__name__
                })
        
        # Remove duplicates, favoring higher weights
        unique_chunks = {}
        for chunk in all_chunks:
            chunk_text = chunk['text']
            if chunk_text not in unique_chunks or chunk['weight'] > unique_chunks[chunk_text]['weight']:
                unique_chunks[chunk_text] = chunk
        
        # Format final chunks with strategy information
        final_chunks = [f"[STRATEGY:{c['strategy']}] {c['text']}" for c in unique_chunks.values()]
        return final_chunks


# Example usage
def demonstrate_chunking_strategies():
    """Demonstrate all chunking strategies with an example text."""
    example_text = """
    # Introduction to Text Chunking
    
    Text chunking is an essential preprocessing step in many natural language processing tasks. 
    It involves breaking down large documents into smaller, more manageable pieces.
    This allows for more efficient processing and analysis of text data.
    
    ## Fixed-Length Chunking
    
    Fixed-length chunking divides text into chunks of a predetermined size.
    This approach is simple but may break semantic units like sentences or paragraphs.
    
    ## Semantic Chunking
    
    Semantic chunking attempts to preserve meaning by keeping related content together.
    It uses techniques like sentence similarity to determine chunk boundaries.
    
    ```python
    def example_code():
        print("This is an example of code that should be kept together")
        for i in range(10):
            print(f"Processing item {i}")