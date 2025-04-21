from typing import List, Dict, Tuple, Optional, Union, Callable
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import udf, col, explode, array, split, size, lit, struct, collect_list
from pyspark.sql.types import ArrayType, StringType, StructType, StructField, IntegerType, FloatType, BinaryType
from pyspark.ml.feature import Tokenizer, CountVectorizer, IDF
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.window import Window
import pyspark.sql.functions as F
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class ChunkingStrategy:
    """Base class for all PySpark chunking strategies."""
    
    def __init__(self, spark: Optional[SparkSession] = None):
        """
        Initialize the chunking strategy.
        
        Args:
            spark: SparkSession instance. If None, will create a new one.
        """
        if spark is None:
            self.spark = SparkSession.builder \
                .appName("TextChunking") \
                .config("spark.driver.memory", "4g") \
                .config("spark.executor.memory", "4g") \
                .getOrCreate()
        else:
            self.spark = spark
        
        # Register UDFs that will be used across strategies
        self._register_common_udfs()
        
        logger.info(f"Initialized {self.__class__.__name__}")
    
    def _register_common_udfs(self):
        """Register common UDFs for text processing."""
        # Register word tokenization UDF
        @udf(returnType=ArrayType(StringType()))
        def word_tokenize_udf(text):
            if not text or not isinstance(text, str):
                return []
            return word_tokenize(text)
        
        # Register sentence tokenization UDF
        @udf(returnType=ArrayType(StringType()))
        def sent_tokenize_udf(text):
            if not text or not isinstance(text, str):
                return []
            return sent_tokenize(text)
        
        # Register paragraph tokenization UDF
        @udf(returnType=ArrayType(StringType()))
        def paragraph_tokenize_udf(text):
            if not text or not isinstance(text, str):
                return []
            paragraphs = re.split(r'\n\s*\n', text)
            return [p.strip() for p in paragraphs if p.strip()]
        
        self.word_tokenize_udf = word_tokenize_udf
        self.sent_tokenize_udf = sent_tokenize_udf
        self.paragraph_tokenize_udf = paragraph_tokenize_udf
    
    def chunk_dataframe(self, df: DataFrame, text_col: str, output_col: str = "chunks") -> DataFrame:
        """
        Apply chunking to a DataFrame column.
        
        Args:
            df: Input DataFrame
            text_col: Column name containing text to chunk
            output_col: Column name for output chunks
            
        Returns:
            DataFrame with chunks added as a new column
        """
        raise NotImplementedError("Subclasses must implement this method")


class FixedLengthChunking(ChunkingStrategy):
    """Chunk text into fixed-length segments using PySpark."""
    
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
        
        logger.info(f"Initialized FixedLengthChunking with chunk_size={chunk_size}, overlap={overlap}")
    
    def chunk_dataframe(self, df: DataFrame, text_col: str, output_col: str = "chunks") -> DataFrame:
        """
        Apply fixed-length chunking to a DataFrame column.
        
        Args:
            df: Input DataFrame
            text_col: Column name containing text to chunk
            output_col: Column name for output chunks
            
        Returns:
            DataFrame with chunks added as a new column
        """
        # First tokenize the text
        df_with_tokens = df.withColumn("tokens", self.word_tokenize_udf(col(text_col)))
        
        # Define UDF for chunking
        @udf(returnType=ArrayType(StringType()))
        def fixed_length_chunk_udf(tokens):
            if not tokens:
                return []
                
            chunks = []
            i = 0
            while i < len(tokens):
                chunk_end = min(i + self.chunk_size, len(tokens))
                chunks.append(" ".join(tokens[i:chunk_end]))
                i += self.chunk_size - self.overlap
                
            return chunks
        
        # Apply chunking
        result_df = df_with_tokens.withColumn(output_col, fixed_length_chunk_udf(col("tokens")))
        
        # Drop intermediate columns
        result_df = result_df.drop("tokens")
        
        return result_df


class SentenceBasedChunking(ChunkingStrategy):
    """Chunk text based on sentences using PySpark."""
    
    def __init__(self, max_sentences: int = 5, **kwargs):
        """
        Initialize sentence-based chunking.
        
        Args:
            max_sentences: Maximum number of sentences per chunk
            **kwargs: Additional arguments for the base class
        """
        super().__init__(**kwargs)
        self.max_sentences = max_sentences
        
        logger.info(f"Initialized SentenceBasedChunking with max_sentences={max_sentences}")
    
    def chunk_dataframe(self, df: DataFrame, text_col: str, output_col: str = "chunks") -> DataFrame:
        """
        Apply sentence-based chunking to a DataFrame column.
        
        Args:
            df: Input DataFrame
            text_col: Column name containing text to chunk
            output_col: Column name for output chunks
            
        Returns:
            DataFrame with chunks added as a new column
        """
        # First tokenize the text into sentences
        df_with_sentences = df.withColumn("sentences", self.sent_tokenize_udf(col(text_col)))
        
        # Define UDF for chunking
        @udf(returnType=ArrayType(StringType()))
        def sentence_chunk_udf(sentences):
            if not sentences:
                return []
                
            chunks = []
            for i in range(0, len(sentences), self.max_sentences):
                chunk = " ".join(sentences[i:i + self.max_sentences])
                chunks.append(chunk)
                
            return chunks
        
        # Apply chunking
        result_df = df_with_sentences.withColumn(output_col, sentence_chunk_udf(col("sentences")))
        
        # Drop intermediate columns
        result_df = result_df.drop("sentences")
        
        return result_df


class ParagraphBasedChunking(ChunkingStrategy):
    """Chunk text based on paragraphs using PySpark."""
    
    def __init__(self, max_paragraphs: int = 1, **kwargs):
        """
        Initialize paragraph-based chunking.
        
        Args:
            max_paragraphs: Maximum number of paragraphs per chunk
            **kwargs: Additional arguments for the base class
        """
        super().__init__(**kwargs)
        self.max_paragraphs = max_paragraphs
        
        logger.info(f"Initialized ParagraphBasedChunking with max_paragraphs={max_paragraphs}")
    
    def chunk_dataframe(self, df: DataFrame, text_col: str, output_col: str = "chunks") -> DataFrame:
        """
        Apply paragraph-based chunking to a DataFrame column.
        
        Args:
            df: Input DataFrame
            text_col: Column name containing text to chunk
            output_col: Column name for output chunks
            
        Returns:
            DataFrame with chunks added as a new column
        """
        # First tokenize the text into paragraphs
        df_with_paragraphs = df.withColumn("paragraphs", self.paragraph_tokenize_udf(col(text_col)))
        
        # Define UDF for chunking
        @udf(returnType=ArrayType(StringType()))
        def paragraph_chunk_udf(paragraphs):
            if not paragraphs:
                return []
                
            chunks = []
            for i in range(0, len(paragraphs), self.max_paragraphs):
                chunk = "\n\n".join(paragraphs[i:i + self.max_paragraphs])
                chunks.append(chunk)
                
            return chunks
        
        # Apply chunking
        result_df = df_with_paragraphs.withColumn(output_col, paragraph_chunk_udf(col("paragraphs")))
        
        # Drop intermediate columns
        result_df = result_df.drop("paragraphs")
        
        return result_df


class SlidingWindowChunking(ChunkingStrategy):
    """Chunk text using a sliding window approach with PySpark."""
    
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
        
        logger.info(f"Initialized SlidingWindowChunking with window_size={window_size}, step_size={step_size}")
    
    def chunk_dataframe(self, df: DataFrame, text_col: str, output_col: str = "chunks") -> DataFrame:
        """
        Apply sliding window chunking to a DataFrame column.
        
        Args:
            df: Input DataFrame
            text_col: Column name containing text to chunk
            output_col: Column name for output chunks
            
        Returns:
            DataFrame with chunks added as a new column
        """
        # First tokenize the text
        df_with_tokens = df.withColumn("tokens", self.word_tokenize_udf(col(text_col)))
        
        # Define UDF for chunking
        @udf(returnType=ArrayType(StringType()))
        def sliding_window_chunk_udf(tokens):
            if not tokens or len(tokens) < self.window_size:
                return [" ".join(tokens)] if tokens else []
                
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
        
        # Apply chunking
        result_df = df_with_tokens.withColumn(output_col, sliding_window_chunk_udf(col("tokens")))
        
        # Drop intermediate columns
        result_df = result_df.drop("tokens")
        
        return result_df


class SemanticChunking(ChunkingStrategy):
    """Chunk text based on semantic similarity using PySpark."""
    
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
        
        logger.info(f"Initialized SemanticChunking with similarity_threshold={similarity_threshold}, max_chunk_size={max_chunk_size}")
    
    def chunk_dataframe(self, df: DataFrame, text_col: str, output_col: str = "chunks") -> DataFrame:
        """
        Apply semantic chunking to a DataFrame column.
        
        Args:
            df: Input DataFrame
            text_col: Column name containing text to chunk
            output_col: Column name for output chunks
            
        Returns:
            DataFrame with chunks added as a new column
        """
        # First tokenize the text into sentences
        df_with_sentences = df.withColumn("sentences", self.sent_tokenize_udf(col(text_col)))
        
        # Explode sentences to process each one
        df_exploded = df_with_sentences.select("*", F.posexplode("sentences").alias("pos", "sentence"))
        
        # Create a unique document ID for each row in the original dataframe
        df_exploded = df_exploded.withColumn("doc_id", F.monotonically_increasing_id())
        
        # Tokenize each sentence
        tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
        df_tokenized = tokenizer.transform(df_exploded)
        
        # Create sentence vectors using TF-IDF
        cv = CountVectorizer(inputCol="words", outputCol="tf")
        cv_model = cv.fit(df_tokenized)
        df_tf = cv_model.transform(df_tokenized)
        
        idf = IDF(inputCol="tf", outputCol="tfidf")
        idf_model = idf.fit(df_tf)
        df_tfidf = idf_model.transform(df_tf)
        
        # Convert to a format suitable for computing similarity
        @udf(returnType=ArrayType(FloatType()))
        def vector_to_array(vector):
            return vector.toArray().tolist()
        
        df_tfidf = df_tfidf.withColumn("tfidf_array", vector_to_array(col("tfidf")))
        
        # Compute similarity between consecutive sentences
        window_spec = Window.partitionBy("doc_id").orderBy("pos")
        
        @udf(returnType=FloatType())
        def cosine_similarity(vec1, vec2):
            if not vec1 or not vec2:
                return 0.0
            
            # Convert to numpy arrays
            a = np.array(vec1)
            b = np.array(vec2)
            
            # Compute cosine similarity
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
                
            return float(np.dot(a, b) / (norm_a * norm_b))
        
        df_tfidf = df_tfidf.withColumn(
            "prev_tfidf", 
            F.lag("tfidf_array").over(window_spec)
        )
        
        df_tfidf = df_tfidf.withColumn(
            "similarity", 
            F.when(col("prev_tfidf").isNotNull(), 
                  cosine_similarity(col("tfidf_array"), col("prev_tfidf"))
            ).otherwise(0.0)
        )
        
        # Determine chunk boundaries
        df_tfidf = df_tfidf.withColumn(
            "new_chunk", 
            F.when(
                (col("similarity") < self.similarity_threshold) | 
                (col("pos") == 0), 
                1
            ).otherwise(0)
        )
        
        # Assign chunk IDs
        df_tfidf = df_tfidf.withColumn(
            "chunk_id", 
            F.sum("new_chunk").over(Window.partitionBy("doc_id").orderBy("pos"))
        )
        
        # Group sentences by chunk ID
        df_chunks = df_tfidf.groupBy("doc_id", "chunk_id").agg(
            F.collect_list("sentence").alias("chunk_sentences")
        )
        
        # Join chunks back to original rows
        @udf(returnType=StringType())
        def join_sentences(sentences):
            return " ".join(sentences)
        
        df_chunks = df_chunks.withColumn("chunk", join_sentences(col("chunk_sentences")))
        
        # Group chunks by document
        df_grouped = df_chunks.groupBy("doc_id").agg(
            F.collect_list("chunk").alias(output_col)
        )
        
        # Join back to original dataframe
        result_df = df.withColumn("doc_id", F.monotonically_increasing_id())
        result_df = result_df.join(df_grouped, "doc_id", "left")
        
        # Handle documents with no chunks
        result_df = result_df.withColumn(
            output_col, 
            F.when(col(output_col).isNotNull(), col(output_col)).otherwise(array(col(text_col)))
        )
        
        # Drop intermediate columns
        result_df = result_df.drop("doc_id", "sentences")
        
        return result_df


class RecursiveChunking(ChunkingStrategy):
    """Recursively chunk text based on document structure using PySpark."""
    
    def __init__(self, max_chunk_size: int = 1000, **kwargs):
        """
        Initialize recursive chunking.
        
        Args:
            max_chunk_size: Maximum size of a chunk in tokens
            **kwargs: Additional arguments for the base class
        """
        super().__init__(**kwargs)
        self.max_chunk_size = max_chunk_size
        
        logger.info(f"Initialized RecursiveChunking with max_chunk_size={max_chunk_size}")
    
    def chunk_dataframe(self, df: DataFrame, text_col: str, output_col: str = "chunks") -> DataFrame:
        """
        Apply recursive chunking to a DataFrame column.
        
        Args:
            df: Input DataFrame
            text_col: Column name containing text to chunk
            output_col: Column name for output chunks
            
        Returns:
            DataFrame with chunks added as a new column
        """
        # Define UDF for recursive chunking
        @udf(returnType=ArrayType(StringType()))
        def recursive_chunk_udf(text):
            if not text or not isinstance(text, str):
                return []
            
            # Helper function to split by headers
            def split_by_headers(text):
                header_pattern = r'(?:\n|^)#+\s+.+?(?=\n|$)'
                splits = re.split(header_pattern, text)
                headers = re.findall(header_pattern, text)
                
                sections = []
                for i in range(len(headers)):
                    if i < len(splits) - 1:
                        sections.append(headers[i] + splits[i+1])
                
                # Add first section if it exists
                if splits[0].strip():
                    sections.insert(0, splits[0])
                    
                return sections
            
            # Helper function to check if chunk is too large
            def is_too_large(text):
                return len(word_tokenize(text)) > self.max_chunk_size
            
            # Recursive function to chunk text
            def chunk_recursively(text, level=0):
                if not is_too_large(text):
                    return [text]
                
                chunks = []
                
                # Level 0: Split by headers
                if level == 0:
                    sections = split_by_headers(text)
                    for section in sections:
                        if is_too_large(section):
                            chunks.extend(chunk_recursively(section, level+1))
                        else:
                            chunks.append(section)
                
                # Level 1: Split by paragraphs
                elif level == 1:
                    paragraphs = re.split(r'\n\s*\n', text)
                    for paragraph in paragraphs:
                        if paragraph.strip() and is_too_large(paragraph):
                            chunks.extend(chunk_recursively(paragraph, level+1))
                        elif paragraph.strip():
                            chunks.append(paragraph)
                
                # Level 2: Split by sentences
                elif level == 2:
                    sentences = sent_tokenize(text)
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
                
                # Level 3: Last resort - fixed length chunking
                else:
                    tokens = word_tokenize(text)
                    for i in range(0, len(tokens), self.max_chunk_size):
                        chunks.append(" ".join(tokens[i:i + self.max_chunk_size]))
                
                return chunks
            
            return chunk_recursively(text)
        
        # Apply chunking
        result_df = df.withColumn(output_col, recursive_chunk_udf(col(text_col)))
        
        return result_df


class ContextEnrichedChunking(ChunkingStrategy):
    """Chunk text while preserving context from surrounding text using PySpark."""
    
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
        
        logger.info(f"Initialized ContextEnrichedChunking with context_size={context_size}")
    
    def chunk_dataframe(self, df: DataFrame, text_col: str, output_col: str = "chunks") -> DataFrame:
        """
        Apply context-enriched chunking to a DataFrame column.
        
        Args:
            df: Input DataFrame
            text_col: Column name containing text to chunk
            output_col: Column name for output chunks
            
        Returns:
            DataFrame with chunks added as a new column
        """
        # First apply base strategy
        df_base_chunks = self.base_strategy.chunk_dataframe(df, text_col, "base_chunks")
        
        # Tokenize the original text
        df_with_tokens = df_base_chunks.withColumn("tokens", self.word_tokenize_udf(col(text_col)))
        
        # Define UDF for adding context
        @udf(returnType=ArrayType(StringType()))
        def add_context_udf(text, base_chunks, tokens):
            if not base_chunks or not tokens:
                return base_chunks if base_chunks else []
            
            enriched_chunks = []
            
            for chunk in base_chunks:
                # Find position of chunk in original text
                chunk_tokens = word_tokenize(chunk)
                chunk_start = None
                
                # Find the starting position of the chunk in the original tokens
                for i in range(len(tokens) - len(chunk_tokens) + 1):
                    if tokens[i:i+len(chunk_tokens)] == chunk_tokens:
                        chunk_start = i
                        break
                
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
        
        # Apply context enrichment
        result_df = df_with_tokens.withColumn(
            output_col, 
            add_context_udf(col(text_col), col("base_chunks"), col("tokens"))
        )
        
        # Drop intermediate columns
        result_df = result_df.drop("base_chunks", "tokens")
        
        return result_df


class ModalitySpecificChunking(ChunkingStrategy):
    """Chunk text based on specific modalities or content types using PySpark."""
    
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
        
        logger.info(f"Initialized ModalitySpecificChunking")
    
    def chunk_dataframe(self, df: DataFrame, text_col: str, output_col: str = "chunks") -> DataFrame:
        """
        Apply modality-specific chunking to a DataFrame column.
        
        Args:
            df: Input DataFrame
            text_col: Column name containing text to chunk
            output_col: Column name for output chunks
            
        Returns:
            DataFrame with chunks added as a new column
        """
        # Define UDF for modality-specific chunking
        @udf(returnType=ArrayType(StringType()))
        def modality_chunk_udf(text):
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
        
        # Apply chunking
        result_df = df.withColumn(output_col, modality_chunk_udf(col(text_col)))
        
        return result_df


class AgenticChunking(ChunkingStrategy):
    """Chunk text based on agent-specific criteria using PySpark."""
    
    def __init__(self, agent_rules: Dict[str, str], **kwargs):
        """
        Initialize agentic chunking.
        
        Args:
            agent_rules: Dictionary mapping agent names to regex patterns
            **kwargs: Additional arguments for the base class
        """
        super().__init__(**kwargs)
        self.agent_rules = agent_rules
        
        logger.info(f"Initialized AgenticChunking with {len(agent_rules)} agents")
    
    def chunk_dataframe(self, df: DataFrame, text_col: str, output_col: str = "chunks") -> DataFrame:
        """
        Apply agentic chunking to a DataFrame column.
        
        Args:
            df: Input DataFrame
            text_col: Column name containing text to chunk
            output_col: Column name for output chunks
            
        Returns:
            DataFrame with chunks added as a new column
        """
        # First tokenize the text into sentences
        df_with_sentences = df.withColumn("sentences", self.sent_tokenize_udf(col(text_col)))
        
        # Define UDF for agentic chunking
        @udf(returnType=ArrayType(StringType()))
        def agentic_chunk_udf(sentences):
            if not sentences:
                return []
            
            agent_chunks = {agent: [] for agent in self.agent_rules.keys()}
            
            # Assign sentences to agents based on regex patterns
            for sentence in sentences:
                for agent, pattern in self.agent_rules.items():
                    if re.search(pattern, sentence, re.IGNORECASE):
                        agent_chunks[agent].append(sentence)
            
            # Create final chunks
            chunks = []
            for agent, agent_sentences in agent_chunks.items():
                if agent_sentences:
                    chunks.append(f"[AGENT:{agent}] {' '.join(agent_sentences)}")
                    
            return chunks
        
        # Apply chunking
        result_df = df_with_sentences.withColumn(output_col, agentic_chunk_udf(col("sentences")))
        
        # Drop intermediate columns
        result_df = result_df.drop("sentences")
        
        return result_df


class SubdocumentChunking(ChunkingStrategy):
    """Chunk text based on subdocument boundaries using PySpark."""
    
    def __init__(self, subdoc_patterns: Dict[str, str], **kwargs):
        """
        Initialize subdocument chunking.
        
        Args:
            subdoc_patterns: Dictionary mapping subdocument types to regex patterns
            **kwargs: Additional arguments for the base class
        """
        super().__init__(**kwargs)
        self.subdoc_patterns = subdoc_patterns
        
        logger.info(f"Initialized SubdocumentChunking with {len(subdoc_patterns)} patterns")
    
    def chunk_dataframe(self, df: DataFrame, text_col: str, output_col: str = "chunks") -> DataFrame:
        """
        Apply subdocument chunking to a DataFrame column.
        
        Args:
            df: Input DataFrame
            text_col: Column name containing text to chunk
            output_col: Column name for output chunks
            
        Returns:
            DataFrame with chunks added as a new column
        """
        # Define UDF for subdocument chunking
        @udf(returnType=ArrayType(StringType()))
        def subdocument_chunk_udf(text):
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
                
            return chunks
        
        # Apply chunking
        result_df = df.withColumn(output_col, subdocument_chunk_udf(col(text_col)))
        
        return result_df


class HybridChunking(ChunkingStrategy):
    """Combine multiple chunking strategies using PySpark."""
    
    def __init__(self, strategies: List[Tuple[ChunkingStrategy, float]], **kwargs):
        """
        Initialize hybrid chunking.
        
        Args:
            strategies: List of (strategy, weight) tuples
            **kwargs: Additional arguments for the base class
        """
        super().__init__(**kwargs)
        self.strategies = strategies
        
        logger.info(f"Initialized HybridChunking with {len(strategies)} strategies")
    
    def chunk_dataframe(self, df: DataFrame, text_col: str, output_col: str = "chunks") -> DataFrame:
        """
        Apply hybrid chunking to a DataFrame column.
        
        Args:
            df: Input DataFrame
            text_col: Column name containing text to chunk
            output_col: Column name for output chunks
            
        Returns:
            DataFrame with chunks added as a new column
        """
        # Apply each strategy and collect results
        strategy_dfs = []
        
        for i, (strategy, weight) in enumerate(self.strategies):
            strategy_name = strategy.__class__.__name__
            strategy_col = f"strategy_{i}_chunks"
            
            # Apply strategy
            strategy_df = strategy.chunk_dataframe(df, text_col, strategy_col)
            
            # Add strategy info and weight
            strategy_df = strategy_df.withColumn(
                f"strategy_{i}_info",
                F.explode(col(strategy_col))
            ).withColumn(
                f"strategy_{i}_with_info",
                F.struct(
                    col(f"strategy_{i}_info").alias("text"),
                    lit(weight).alias("weight"),
                    lit(strategy_name).alias("strategy")
                )
            )
            
            strategy_dfs.append(strategy_df.select("*", F.collect_list(f"strategy_{i}_with_info").alias(f"strategy_{i}_all")))
        
        # Join all strategy results
        result_df = df
        for i in range(len(self.strategies)):
            result_df = result_df.join(
                strategy_dfs[i].select(F.monotonically_increasing_id().alias("id"), f"strategy_{i}_all"),
                F.monotonically_increasing_id() == col("id"),
                "left"
            ).drop("id")
        
        # Combine all strategies
        @udf(returnType=ArrayType(StringType()))
        def combine_strategies(*strategy_arrays):
            all_chunks = []
            
            # Collect all chunks from all strategies
            for strategy_array in strategy_arrays:
                if strategy_array:
                    all_chunks.extend(strategy_array)
            
            # Remove duplicates, favoring higher weights
            unique_chunks = {}
            for chunk in all_chunks:
                chunk_text = chunk["text"]
                if chunk_text not in unique_chunks or chunk["weight"] > unique_chunks[chunk_text]["weight"]:
                    unique_chunks[chunk_text] = chunk
            
            # Format final chunks with strategy information
            final_chunks = [f"[STRATEGY:{c['strategy']}] {c['text']}" for c in unique_chunks.values()]
            return final_chunks
        
        # Combine all strategies
        strategy_cols = [col(f"strategy_{i}_all") for i in range(len(self.strategies))]
        result_df = result_df.withColumn(output_col, combine_strategies(*strategy_cols))
        
        # Drop intermediate columns
        for i in range(len(self.strategies)):
            result_df = result_df.drop(f"strategy_{i}_all")
        
        return result_df


# Example usage
def demonstrate_chunking_strategies():
    """Demonstrate all chunking strategies with an example text."""
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("ChunkingDemo") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    
    # Example text
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
    ```
    
    | Strategy | Pros | Cons |
    |----------|------|------|
    | Fixed-Length | Simple, consistent | May break semantic units |
    | Semantic | Preserves meaning | More complex, computationally expensive |
    | Hybrid | Flexible, adaptable | Requires careful tuning |
    
    1. First advantage of chunking
    2. Second advantage of chunking
    3. Third advantage of chunking
    
    The equation for calculating optimal chunk size might be: $C = \frac{D}{N} \times f$
    
    In conclusion, choosing the right chunking strategy depends on your specific use case and requirements.
    """
    
    # Create a sample dataframe
    data = [("doc1", example_text), ("doc2", example_text[:500])]
    df = spark.createDataFrame(data, ["doc_id", "text"])
    
    # Initialize strategies
    fixed_length = FixedLengthChunking(chunk_size=50, overlap=10, spark=spark)
    sentence_based = SentenceBasedChunking(max_sentences=2, spark=spark)
    paragraph_based = ParagraphBasedChunking(spark=spark)
    sliding_window = SlidingWindowChunking(window_size=50, step_size=25, spark=spark)
    semantic = SemanticChunking(spark=spark)
    recursive = RecursiveChunking(spark=spark)
    context_enriched = ContextEnrichedChunking(base_strategy=sentence_based, context_size=20, spark=spark)
    
    modality_specific = ModalitySpecificChunking(spark=spark)
    
    # Define agent rules for agentic chunking
    agent_rules = {
        'code_agent': r'code|python',
        'math_agent': r'equation|calculate',
        'summary_agent': r'introduction|conclusion'
    }
    agentic = AgenticChunking(agent_rules=agent_rules, spark=spark)
    
    # Define subdocument patterns
    subdoc_patterns = {
        'header': r'(?:\n|^)#+\s+.+?(?=\n|$)',
        'code_block': r'```[\s\S]*?```',
        'table': r'\|.*\|[\s]*\n\|[\s]*[-:]+[-|\s:]*\n(\|.*\|[\s]*\n)+'
    }
    subdocument = SubdocumentChunking(subdoc_patterns=subdoc_patterns, spark=spark)
    
    # Define hybrid strategy
    hybrid = HybridChunking(strategies=[
        (sentence_based, 0.5),
        (modality_specific, 0.7)
    ], spark=spark)
    
    # Demonstrate each strategy
    strategies = {
        "Fixed-Length Chunking": fixed_length,
        "Sentence-Based Chunking": sentence_based,
        "Paragraph-Based Chunking": paragraph_based,
        "Sliding Window Chunking": sliding_window,
        "Semantic Chunking": semantic,
        "Recursive Chunking": recursive,
        "Context-Enriched Chunking": context_enriched,
        "Modality-Specific Chunking": modality_specific,
        "Agentic Chunking": agentic,
        "Subdocument Chunking": subdocument,
        "Hybrid Chunking": hybrid
    }
    
    for name, strategy in strategies.items():
        print(f"\n=== {name} ===")
        result_df = strategy.chunk_dataframe(df, "text", "chunks")
        
        # Show chunks for the first document
        chunks = result_df.filter(col("doc_id") == "doc1").select(explode(col("chunks")).alias("chunk")).collect()
        for i, row in enumerate(chunks):
            chunk = row["chunk"]
            print(f"Chunk {i+1}: {chunk[:100]}..." if len(chunk) > 100 else f"Chunk {i+1}: {chunk}")
    
    spark.stop()

# Run the demonstration if executed directly
if __name__ == "__main__":
    demonstrate_chunking_strategies()