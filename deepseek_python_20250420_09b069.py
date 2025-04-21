from pyspark.sql import DataFrame
from pyspark.sql.functions import col, udf, explode, length, collect_list
from pyspark.sql.types import ArrayType, StringType, StructType, StructField, IntegerType
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import List, Callable, Optional
from abc import ABC, abstractmethod

nltk.download('punkt', quiet=True)

# Define common schema for chunk metadata
CHUNK_SCHEMA = StructType([
    StructField("text", StringType()),
    StructField("start", IntegerType()),
    StructField("end", IntegerType())
])

class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies"""
    
    @abstractmethod
    def chunk(self, df: DataFrame, text_col: str = "text", chunk_col: str = "chunks") -> DataFrame:
        pass

class FixedLengthChunking(ChunkingStrategy):
    """Fixed-length chunking with optional overlap"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 0):
        self.chunk_size = chunk_size
        self.overlap = max(0, min(overlap, chunk_size - 1))

    def chunk(self, df: DataFrame, text_col: str = "text", chunk_col: str = "chunks") -> DataFrame:
        def split_text(text: str) -> List[List]:
            chunks = []
            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunk = text[start:end]
                chunks.append((chunk, start, end-1))
                start = end - self.overlap
            return chunks
        
        split_udf = udf(split_text, ArrayType(CHUNK_SCHEMA))
        return df.withColumn(chunk_col, split_udf(col(text_col)))

class SentenceBasedChunking(ChunkingStrategy):
    """Sentence-based chunking using NLTK"""
    
    def chunk(self, df: DataFrame, text_col: str = "text", chunk_col: str = "chunks") -> DataFrame:
        def split_sentences(text: str) -> List[List]:
            sentences = nltk.sent_tokenize(text)
            chunks = []
            pos = 0
            for sent in sentences:
                start = text.find(sent, pos)
                end = start + len(sent)
                chunks.append((sent, start, end-1))
                pos = end
            return chunks
        
        sentence_udf = udf(split_sentences, ArrayType(CHUNK_SCHEMA))
        return df.withColumn(chunk_col, sentence_udf(col(text_col)))

class SlidingWindowChunking(ChunkingStrategy):
    """Sliding window chunking with overlap"""
    
    def __init__(self, window_size: int = 512, step_size: int = 256):
        self.window_size = window_size
        self.step_size = step_size

    def chunk(self, df: DataFrame, text_col: str = "text", chunk_col: str = "chunks") -> DataFrame:
        def sliding_window(text: str) -> List[List]:
            chunks = []
            start = 0
            while start < len(text):
                end = start + self.window_size
                chunk = text[start:end]
                chunks.append((chunk, start, end-1))
                start += self.step_size
            return chunks
        
        window_udf = udf(sliding_window, ArrayType(CHUNK_SCHEMA))
        return df.withColumn(chunk_col, window_udf(col(text_col)))

class SemanticChunking(ChunkingStrategy):
    """Semantic chunking using sentence embeddings"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', threshold: float = 0.85):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold

    def chunk(self, df: DataFrame, text_col: str = "text", chunk_col: str = "chunks") -> DataFrame:
        sentence_chunker = SentenceBasedChunking()
        sentence_df = sentence_chunker.chunk(df, text_col, "sentences")
        
        def process_sentences(sentences: List[str]) -> List[List]:
            if not sentences:
                return []
            embeddings = self.model.encode(sentences)
            chunks = []
            current_chunk = []
            start_pos = 0
            end_pos = 0
            
            for i in range(len(sentences)-1):
                current_chunk.append(sentences[i])
                similarity = cosine_similarity(
                    [embeddings[i]], [embeddings[i+1]]
                )[0][0]
                
                if similarity < self.threshold:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append((chunk_text, start_pos, end_pos + len(sentences[i])))
                    start_pos = end_pos + len(sentences[i]) + 1
                    current_chunk = []
                end_pos += len(sentences[i]) + 1
            
            # Add remaining text
            if current_chunk or len(sentences) == 1:
                current_chunk.extend(sentences[len(current_chunk):])
                chunk_text = ' '.join(current_chunk)
                chunks.append((chunk_text, start_pos, end_pos + len(sentences[-1])))
            
            return chunks
        
        process_udf = udf(process_sentences, ArrayType(CHUNK_SCHEMA))
        return sentence_df.withColumn(chunk_col, process_udf(col("sentences"))).drop("sentences")

class HybridChunking(ChunkingStrategy):
    """Hybrid chunking combining multiple strategies"""
    
    def __init__(self, primary_strategy: ChunkingStrategy, 
                 secondary_strategy: ChunkingStrategy, 
                 max_length: int = 1024):
        self.primary = primary_strategy
        self.secondary = secondary_strategy
        self.max_length = max_length

    def chunk(self, df: DataFrame, text_col: str = "text", chunk_col: str = "chunks") -> DataFrame:
        primary_df = self.primary.chunk(df, text_col, "primary_chunks")
        
        def process_chunks(chunks: List) -> List:
            final_chunks = []
            for chunk in chunks:
                if chunk.end - chunk.start > self.max_length:
                    sub_chunks = self.secondary.chunk(
                        chunk.text, "text", "secondary_chunks"
                    ).collect()[0]["secondary_chunks"]
                    final_chunks.extend(sub_chunks)
                else:
                    final_chunks.append(chunk)
            return final_chunks
        
        process_udf = udf(process_chunks, ArrayType(CHUNK_SCHEMA))
        return primary_df.withColumn(chunk_col, process_udf(col("primary_chunks"))).drop("primary_chunks")

class ChunkerFactory:
    """Factory class for creating chunking strategies"""
    
    @staticmethod
    def create_chunker(strategy: str, **kwargs) -> ChunkingStrategy:
        strategies = {
            "fixed_length": FixedLengthChunking,
            "sentence": SentenceBasedChunking,
            "sliding_window": SlidingWindowChunking,
            "semantic": SemanticChunking,
            "hybrid": HybridChunking
        }
        
        if strategy not in strategies:
            raise ValueError(f"Unsupported strategy: {strategy}")
            
        return strategies[strategy](**kwargs)

# Example usage
if __name__ == "__main__":
    from pyspark.sql import SparkSession
    
    spark = SparkSession.builder \
        .appName("ChunkingExample") \
        .getOrCreate()
    
    # Sample data
    data = [("1", "This is a sample text. It contains multiple sentences. " 
                  "We will use it to demonstrate chunking strategies.")]
    df = spark.createDataFrame(data, ["id", "text"])
    
    # Create chunker
    factory = ChunkerFactory()
    chunker = factory.create_chunker("hybrid",
        primary_strategy=FixedLengthChunking(100),
        secondary_strategy=SentenceBasedChunking(),
        max_length=100
    )
    
    # Apply chunking
    chunked_df = chunker.chunk(df)
    chunked_df.show(truncate=False)