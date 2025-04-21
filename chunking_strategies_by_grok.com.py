from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, udf, explode, length, split
from pyspark.sql.types import ArrayType, StringType, StructType, StructField
from typing import List, Optional
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import logging

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChunkingStrategy:
    """Abstract base class for chunking strategies."""
    def chunk(self, df: DataFrame, text_column: str) -> DataFrame:
        raise NotImplementedError("Chunk method must be implemented by subclasses.")

class FixedLengthChunking(ChunkingStrategy):
    """Chunks text into fixed-length segments."""
    def __init__(self, chunk_size: int = 100):
        self.chunk_size = chunk_size

    def chunk(self, df: DataFrame, text_column: str) -> DataFrame:
        """Splits text into fixed-length chunks."""
        try:
            @udf(ArrayType(StringType()))
            def fixed_length_split(text: str) -> List[str]:
                return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]

            return (df
                    .withColumn("chunks", fixed_length_split(col(text_column)))
                    .withColumn("chunk", explode(col("chunks")))
                    .drop("chunks"))
        except Exception as e:
            logger.error(f"Error in FixedLengthChunking: {str(e)}")
            raise

class SentenceBasedChunking(ChunkingStrategy):
    """Chunks text into sentences."""
    def chunk(self, df: DataFrame, text_column: str) -> DataFrame:
        """Splits text into individual sentences."""
        try:
            @udf(ArrayType(StringType()))
            def sentence_split(text: str) -> List[str]:
                return sent_tokenize(text)

            return (df
                    .withColumn("chunks", sentence_split(col(text_column)))
                    .withColumn("chunk", explode(col("chunks")))
                    .drop("chunks"))
        except Exception as e:
            logger.error(f"Error in SentenceBasedChunking: {str(e)}")
            raise

class ParagraphBasedChunking(ChunkingStrategy):
    """Chunks text into paragraphs."""
    def chunk(self, df: DataFrame, text_column: str) -> DataFrame:
        """Splits text into paragraphs based on double newlines."""
        try:
            @udf(ArrayType(StringType()))
            def paragraph_split(text: str) -> List[str]:
                return [p.strip() for p in text.split('\n\n') if p.strip()]

            return (df
                    .withColumn("chunks", paragraph_split(col(text_column)))
                    .withColumn("chunk", explode(col("chunks")))
                    .drop("chunks"))
        except Exception as e:
            logger.error(f"Error in ParagraphBasedChunking: {str(e)}")
            raise

class SlidingWindowChunking(ChunkingStrategy):
    """Chunks text using a sliding window approach."""
    def __init__(self, window_size: int = 100, step_size: int = 50):
        self.window_size = window_size
        self.step_size = step_size

    def chunk(self, df: DataFrame, text_column: str) -> DataFrame:
        """Splits text into overlapping windows."""
        try:
            @udf(ArrayType(StringType()))
            def sliding_window(text: str) -> List[str]:
                chunks = []
                for i in range(0, len(text) - self.window_size + 1, self.step_size):
                    chunks.append(text[i:i+self.window_size])
                return chunks

            return (df
                    .withColumn("chunks", sliding_window(col(text_column)))
                    .withColumn("chunk", explode(col("chunks")))
                    .drop("chunks"))
        except Exception as e:
            logger.error(f"Error in SlidingWindowChunking: {str(e)}")
            raise

class SemanticChunking(ChunkingStrategy):
    """Chunks text based on semantic similarity (simplified using keyword density)."""
    def __init__(self, max_chunk_size: int = 200):
        self.max_chunk_size = max_chunk_size
        self.stop_words = set(stopwords.words('english'))

    def chunk(self, df: DataFrame, text_column: str) -> DataFrame:
        """Splits text based on semantic boundaries."""
        try:
            @udf(ArrayType(StringType()))
            def semantic_split(text: str) -> List[str]:
                sentences = sent_tokenize(text)
                chunks = []
                current_chunk = []
                current_length = 0

                for sentence in sentences:
                    if current_length + len(sentence) <= self.max_chunk_size:
                        current_chunk.append(sentence)
                        current_length += len(sentence)
                    else:
                        chunks.append(' '.join(current_chunk))
                        current_chunk = [sentence]
                        current_length = len(sentence)
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                return chunks

            return (df
                    .withColumn("chunks", semantic_split(col(text_column)))
                    .withColumn("chunk", explode(col("chunks")))
                    .drop("chunks"))
        except Exception as e:
            logger.error(f"Error in SemanticChunking: {str(e)}")
            raise

class RecursiveChunking(ChunkingStrategy):
    """Recursively chunks text into smaller units."""
    def __init__(self, max_chunk_size: int = 500, min_chunk_size: int = 100):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def chunk(self, df: DataFrame, text_column: str) -> DataFrame:
        """Recursively splits text into smaller chunks."""
        try:
            @udf(ArrayType(StringType()))
            def recursive_split(text: str) -> List[str]:
                def split_recursive(text: str, chunks: List[str]) -> List[str]:
                    if len(text) <= self.max_chunk_size:
                        if len(text) >= self.min_chunk_size:
                            chunks.append(text)
                        return chunks
                    mid = len(text) // 2
                    split_point = text.rfind(' ', 0, mid)
                    if split_point == -1:
                        split_point = mid
                    chunks = split_recursive(text[:split_point], chunks)
                    chunks = split_recursive(text[split_point:], chunks)
                    return chunks

                return split_recursive(text, [])

            return (df
                    .withColumn("chunks", recursive_split(col(text_column)))
                    .withColumn("chunk", explode(col("chunks")))
                    .drop("chunks"))
        except Exception as e:
            logger.error(f"Error in RecursiveChunking: {str(e)}")
            raise

class ContextEnrichedChunking(ChunkingStrategy):
    """Enriches chunks with contextual metadata."""
    def __init__(self, chunk_size: int = 200):
        self.chunk_size = chunk_size

    def chunk(self, df: DataFrame, text_column: str) -> DataFrame:
        """Splits text into chunks with contextual metadata."""
        try:
            @udf(ArrayType(StructType([
                StructField("chunk", StringType()),
                StructField("context", StringType())
            ])))
            def context_split(text: str) -> List[dict]:
                chunks = [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]
                return [{"chunk": chunk, "context": f"Position: {i//self.chunk_size}"} 
                        for i, chunk in enumerate(chunks)]

            return (df
                    .withColumn("chunks", context_split(col(text_column)))
                    .withColumn("chunk_struct", explode(col("chunks")))
                    .select(col(text_column), col("chunk_struct.chunk"), col("chunk_struct.context")))
        except Exception as e:
            logger.error(f"Error in ContextEnrichedChunking: {str(e)}")
            raise

class ModalitySpecificChunking(ChunkingStrategy):
    """Chunks text based on modality (e.g., code blocks, prose)."""
    def chunk(self, df: DataFrame, text_column: str) -> DataFrame:
        """Splits text based on modality patterns."""
        try:
            @udf(ArrayType(StringType()))
            def modality_split(text: str) -> List[str]:
                # Simple regex to detect code blocks
                code_pattern = r'```[\s\S]*?```'
                chunks = []
                last_pos = 0
                
                for match in re.finditer(code_pattern, text):
                    start, end = match.span()
                    if last_pos < start:
                        chunks.append(text[last_pos:start])
                    chunks.append(text[start:end])
                    last_pos = end
                
                if last_pos < len(text):
                    chunks.append(text[last_pos:])
                return [c.strip() for c in chunks if c.strip()]

            return (df
                    .withColumn("chunks", modality_split(col(text_column)))
                    .withColumn("chunk", explode(col("chunks")))
                    .drop("chunks"))
        except Exception as e:
            logger.error(f"Error in ModalitySpecificChunking: {str(e)}")
            raise

class AgenticChunking(ChunkingStrategy):
    """Intelligent chunking based on content analysis."""
    def __init__(self, max_chunk_size: int = 300):
        self.max_chunk_size = max_chunk_size

    def chunk(self, df: DataFrame, text_column: str) -> DataFrame:
        """Splits text based on content importance."""
        try:
            @udf(ArrayType(StringType()))
            def agentic_split(text: str) -> List[str]:
                sentences = sent_tokenize(text)
                chunks = []
                current_chunk = []
                current_length = 0

                for sentence in sentences:
                    if current_length + len(sentence) <= self.max_chunk_size:
                        current_chunk.append(sentence)
                        current_length += len(sentence)
                    else:
                        chunks.append(' '.join(current_chunk))
                        current_chunk = [sentence]
                        current_length = len(sentence)
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                return chunks

            return (df
                    .withColumn("chunks", agentic_split(col(text_column)))
                    .withColumn("chunk", explode(col("chunks")))
                    .drop("chunks"))
        except Exception as e:
            logger.error(f"Error in AgenticChunking: {str(e)}")
            raise

class SubdocumentChunking(ChunkingStrategy):
    """Chunks text into subdocuments based on headers."""
    def chunk(self, df: DataFrame, text_column: str) -> DataFrame:
        """Splits text based on header patterns."""
        try:
            @udf(ArrayType(StringType()))
            def subdocument_split(text: str) -> List[str]:
                # Simple header detection
                header_pattern = r'^(#+ .+)$'
                chunks = []
                current_chunk = []
                lines = text.split('\n')

                for line in lines:
                    if re.match(header_pattern, line):
                        if current_chunk:
                            chunks.append('\n'.join(current_chunk))
                            current_chunk = [line]
                        else:
                            current_chunk.append(line)
                    else:
                        current_chunk.append(line)
                
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                return [c.strip() for c in chunks if c.strip()]

            return (df
                    .withColumn("chunks", subdocument_split(col(text_column)))
                    .withColumn("chunk", explode(col("chunks")))
                    .drop("chunks"))
        except Exception as e:
            logger.error(f"Error in SubdocumentChunking: {str(e)}")
            raise

class HybridChunking(ChunkingStrategy):
    """Combines multiple chunking strategies."""
    def __init__(self, chunk_size: int = 200, window_size: int = 100, step_size: int = 50):
        self.fixed_chunker = FixedLengthChunking(chunk_size)
        self.sliding_chunker = SlidingWindowChunking(window_size, step_size)
        self.sentence_chunker = SentenceBasedChunking()

    def chunk(self, df: DataFrame, text_column: str) -> DataFrame:
        """Applies multiple chunking strategies sequentially."""
        try:
            # First apply sentence-based chunking
            df_sentences = self.sentence_chunker.chunk(df, text_column)
            
            # Then apply fixed-length chunking
            df_fixed = self.fixed_chunker.chunk(df_sentences, "chunk")
            
            # Finally apply sliding window
            df_final = self.sliding_chunker.chunk(df_fixed, "chunk")
            
            return df_final
        except Exception as e:
            logger.error(f"Error in HybridChunking: {str(e)}")
            raise

def main():
    """Example usage of chunking strategies."""
    spark = SparkSession.builder.appName("ChunkingStrategies").getOrCreate()
    
    # Sample data
    data = [("1", "This is a sample text. It contains multiple sentences. Here is another paragraph.\n\nSecond paragraph.")]
    df = spark.createDataFrame(data, ["id", "text"])
    
    # Test each chunking strategy
    strategies = [
        FixedLengthChunking(chunk_size=10),
        SentenceBasedChunking(),
        ParagraphBasedChunking(),
        SlidingWindowChunking(window_size=20, step_size=10),
        SemanticChunking(max_chunk_size=50),
        RecursiveChunking(max_chunk_size=50, min_chunk_size=10),
        ContextEnrichedChunking(chunk_size=20),
        ModalitySpecificChunking(),
        AgenticChunking(max_chunk_size=50),
        SubdocumentChunking(),
        HybridChunking(chunk_size=20, window_size=15, step_size=5)
    ]
    
    for strategy in strategies:
        logger.info(f"Applying {strategy.__class__.__name__}")
        result = strategy.chunk(df, "text")
        result.show(truncate=False)

if __name__ == "__main__":
    main()