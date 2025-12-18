"""
RAG Engine for HR FAQ Chatbot
Retrieval-Augmented Generation using ChromaDB and Sentence Transformers
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import hashlib

# Lazy imports to handle missing dependencies gracefully
chromadb = None
SentenceTransformer = None


def _load_dependencies():
    """Load optional dependencies"""
    global chromadb, SentenceTransformer

    if chromadb is None:
        try:
            import chromadb as _chromadb

            chromadb = _chromadb
        except ImportError:
            print("Warning: chromadb not installed. Run: pip install chromadb")

    if SentenceTransformer is None:
        try:
            from sentence_transformers import SentenceTransformer as _ST

            SentenceTransformer = _ST
        except ImportError:
            print(
                "Warning: sentence-transformers not installed. Run: pip install sentence-transformers"
            )


class DocumentChunk:
    """Represents a chunk of a document"""

    def __init__(
        self,
        content: str,
        source_file: str,
        section: str,
        chunk_id: str,
        metadata: Optional[Dict] = None,
    ):
        self.content = content
        self.source_file = source_file
        self.section = section
        self.chunk_id = chunk_id
        self.metadata = metadata or {}

    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "source_file": self.source_file,
            "section": self.section,
            "chunk_id": self.chunk_id,
            **self.metadata,
        }


class MarkdownParser:
    """Parse markdown documents into chunks"""

    @staticmethod
    def parse_file(file_path: str) -> List[DocumentChunk]:
        """Parse a markdown file into document chunks"""
        chunks = []

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract document title
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        doc_title = title_match.group(1) if title_match else Path(file_path).stem

        # Split by headers (## or ###)
        sections = re.split(r"\n(?=##\s)", content)

        for i, section in enumerate(sections):
            if not section.strip():
                continue

            # Extract section title
            section_title_match = re.search(r"^##\s*(.+?)(?:\n|$)", section)
            section_title = (
                section_title_match.group(1).strip()
                if section_title_match
                else f"Section {i}"
            )

            # Clean the content
            section_content = section.strip()

            # Skip very short sections
            if len(section_content) < 50:
                continue

            # Create chunk ID based on content hash
            chunk_id = hashlib.md5(
                f"{file_path}:{section_title}:{i}".encode()
            ).hexdigest()[:12]

            # Determine category from file path
            category = "general"
            if "policies" in file_path.lower():
                category = "policy"
            elif "benefits" in file_path.lower():
                category = "benefits"
            elif "payroll" in file_path.lower():
                category = "payroll"
            elif "onboarding" in file_path.lower():
                category = "onboarding"

            chunk = DocumentChunk(
                content=section_content,
                source_file=file_path,
                section=f"{doc_title} - {section_title}",
                chunk_id=chunk_id,
                metadata={
                    "category": category,
                    "doc_title": doc_title,
                    "section_title": section_title,
                },
            )
            chunks.append(chunk)

        return chunks

    @staticmethod
    def parse_directory(directory: str) -> List[DocumentChunk]:
        """Parse all markdown files in a directory"""
        chunks = []
        directory_path = Path(directory)

        if not directory_path.exists():
            print(f"Warning: Directory {directory} does not exist")
            return chunks

        for md_file in directory_path.rglob("*.md"):
            try:
                file_chunks = MarkdownParser.parse_file(str(md_file))
                chunks.extend(file_chunks)
                print(f"Parsed {len(file_chunks)} chunks from {md_file.name}")
            except Exception as e:
                print(f"Error parsing {md_file}: {e}")

        return chunks


class RAGEngine:
    """
    Retrieval-Augmented Generation Engine
    Uses ChromaDB for vector storage and Sentence Transformers for embeddings
    """

    def __init__(
        self,
        company_data_path: str = "company_data/techcorp_solutions",
        persist_directory: str = "backend/chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2",
        collection_name: str = "hr_documents",
    ):
        _load_dependencies()

        self.company_data_path = company_data_path
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model
        self.collection_name = collection_name

        self.embedder = None
        self.client = None
        self.collection = None
        self.is_initialized = False

        # Company info
        self.company_info = self._load_company_info()

    def _load_company_info(self) -> Dict:
        """Load company metadata"""
        info_path = os.path.join(self.company_data_path, "company_info.json")
        if os.path.exists(info_path):
            with open(info_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"company_name": "TechCorp Solutions"}

    def initialize(self, force_rebuild: bool = False) -> bool:
        """Initialize the RAG engine"""
        if chromadb is None or SentenceTransformer is None:
            print("RAG dependencies not available. Using fallback mode.")
            return False

        try:
            print("Initializing RAG engine...")

            # Load embedding model
            print(f"Loading embedding model: {self.embedding_model_name}")
            self.embedder = SentenceTransformer(self.embedding_model_name)

            # Initialize ChromaDB
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            # Check if collection exists and has data
            try:
                self.collection = self.client.get_collection(self.collection_name)
                doc_count = self.collection.count()
                print(f"Found existing collection with {doc_count} documents")

                if force_rebuild or doc_count == 0:
                    print("Rebuilding collection...")
                    self.client.delete_collection(self.collection_name)
                    self._create_and_populate_collection()
            except Exception:
                print("Creating new collection...")
                self._create_and_populate_collection()

            self.is_initialized = True
            print("RAG engine initialized successfully!")
            return True

        except Exception as e:
            print(f"Error initializing RAG engine: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _create_and_populate_collection(self):
        """Create collection and populate with documents"""
        self.collection = self.client.create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )

        # Parse documents
        chunks = MarkdownParser.parse_directory(self.company_data_path)

        if not chunks:
            print("Warning: No documents found to index")
            return

        print(f"Indexing {len(chunks)} document chunks...")

        # Prepare data for ChromaDB
        documents = []
        metadatas = []
        ids = []

        for chunk in chunks:
            documents.append(chunk.content)
            metadatas.append(
                {
                    "source_file": chunk.source_file,
                    "section": chunk.section,
                    "category": chunk.metadata.get("category", "general"),
                    "doc_title": chunk.metadata.get("doc_title", ""),
                    "section_title": chunk.metadata.get("section_title", ""),
                }
            )
            ids.append(chunk.chunk_id)

        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.embedder.encode(documents, show_progress_bar=True).tolist()

        # Add to collection
        self.collection.add(
            documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids
        )

        print(f"Indexed {len(chunks)} chunks successfully!")

    def search(
        self, query: str, top_k: int = 5, category_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Search for relevant documents

        Args:
            query: The search query
            top_k: Number of results to return
            category_filter: Optional category to filter by (policy, benefits, payroll, onboarding)

        Returns:
            List of relevant document chunks with metadata
        """
        if not self.is_initialized:
            return []

        try:
            # Generate query embedding
            query_embedding = self.embedder.encode([query]).tolist()

            # Build where filter
            where_filter = None
            if category_filter:
                where_filter = {"category": category_filter}

            # Search
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )

            # Format results
            formatted_results = []
            if results and results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    metadata = (
                        results["metadatas"][0][i] if results["metadatas"] else {}
                    )
                    distance = results["distances"][0][i] if results["distances"] else 0

                    # Convert distance to similarity score (cosine distance to similarity)
                    similarity = 1 - distance

                    formatted_results.append(
                        {
                            "content": doc,
                            "source": metadata.get("source_file", "Unknown"),
                            "section": metadata.get("section", ""),
                            "category": metadata.get("category", "general"),
                            "doc_title": metadata.get("doc_title", ""),
                            "section_title": metadata.get("section_title", ""),
                            "similarity": round(similarity, 3),
                        }
                    )

            return formatted_results

        except Exception as e:
            print(f"Error searching: {e}")
            return []

    def get_context_for_question(
        self, question: str, mode: str = "policy", max_context_length: int = 2000
    ) -> Tuple[str, List[Dict]]:
        """
        Get relevant context for a question

        Args:
            question: The user's question
            mode: The mode (policy, benefits, payroll)
            max_context_length: Maximum characters of context to return

        Returns:
            Tuple of (context_string, sources_list)
        """
        # Map mode to category
        category_map = {
            "policy": None,  # Search all for policy mode
            "benefits": "benefits",
            "payroll": "payroll",
        }
        category = category_map.get(mode)

        # Search for relevant documents
        results = self.search(question, top_k=5, category_filter=category)

        if not results:
            return "", []

        # Build context string
        context_parts = []
        sources = []
        current_length = 0

        for result in results:
            content = result["content"]

            # Check if adding this would exceed limit
            if current_length + len(content) > max_context_length:
                # Truncate content to fit
                remaining = max_context_length - current_length
                if remaining > 200:  # Only add if meaningful content
                    content = content[:remaining] + "..."
                else:
                    break

            context_parts.append(f"[{result['section']}]\n{content}")
            current_length += len(content)

            sources.append(
                {
                    "title": result["section"],
                    "snippet": content[:200] + "..." if len(content) > 200 else content,
                    "category": result["category"],
                    "similarity": result["similarity"],
                }
            )

        context = "\n\n---\n\n".join(context_parts)

        return context, sources

    def get_company_name(self) -> str:
        """Get the company name"""
        return self.company_info.get("company_name", "TechCorp Solutions")


# Singleton instance
_rag_engine: Optional[RAGEngine] = None


def get_rag_engine(
    company_data_path: str = "company_data/techcorp_solutions", force_init: bool = False
) -> RAGEngine:
    """Get or create the RAG engine singleton"""
    global _rag_engine

    if _rag_engine is None or force_init:
        _rag_engine = RAGEngine(company_data_path=company_data_path)
        _rag_engine.initialize()

    return _rag_engine


if __name__ == "__main__":
    # Test the RAG engine
    print("Testing RAG Engine...")

    engine = get_rag_engine(force_init=True)

    if engine.is_initialized:
        # Test queries
        test_queries = [
            "How many vacation days do I get?",
            "What is the remote work policy?",
            "How does the 401k matching work?",
            "What training is required for new hires?",
            "How do I report harassment?",
        ]

        for query in test_queries:
            print(f"\n{'=' * 60}")
            print(f"Query: {query}")
            print(f"{'=' * 60}")

            context, sources = engine.get_context_for_question(query)

            print(f"\nFound {len(sources)} relevant sources:")
            for i, source in enumerate(sources, 1):
                print(f"  {i}. {source['title']} (similarity: {source['similarity']})")

            print("\nContext preview (first 500 chars):")
            print(context[:500] + "..." if len(context) > 500 else context)
    else:
        print("RAG engine not initialized. Install dependencies:")
        print("  pip install chromadb sentence-transformers")
