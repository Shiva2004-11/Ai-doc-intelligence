import os
import json
import uuid
import sqlite3
import tempfile
from datetime import datetime
import logging
from collections import Counter
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import gradio as gr
import pdfplumber
import docx
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from dateutil import parser as date_parser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
from sklearn.metrics import accuracy_score
try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_AVAILABLE = True
except:
    SentenceTransformer = None
    util = None
    SENTENCE_AVAILABLE = False

# Import heavy ML libraries with error handling
try:
    # Try to import bertopic but handle potential torch issues
    from bertopic import BERTopic
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False
    logging.warning("BERTopic not installed. Topic modeling disabled.")
except Exception as e:
    BERTOPIC_AVAILABLE = False
    logging.warning(f"BERTopic import failed: {e}. Topic modeling disabled.")

try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
    except:
        # Try to download if not available
        try:
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            nlp = spacy.load("en_core_web_sm")
            SPACY_AVAILABLE = True
        except:
            SPACY_AVAILABLE = False
            logging.warning("spaCy not installed. Knowledge Graph disabled.")
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not installed. Knowledge Graph disabled.")
# ================= CHROMADB INITIALIZATION (FIXED) =================

collection = None
CHROMADB_AVAILABLE = False

try:
    import chromadb
    from chromadb.utils import embedding_functions

    print("✅ ChromaDB module imported")

    # Create persistent client
    chroma_client = chromadb.PersistentClient(path="./chroma_db")

    print("✅ ChromaDB PersistentClient created")

    if SENTENCE_AVAILABLE:
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    else:
        embedding_fn = None

    collection = chroma_client.get_or_create_collection(
        name="enterprise_docs",
        embedding_function=embedding_fn
    )

    CHROMADB_AVAILABLE = True
    print("✅ ChromaDB initialized successfully")

except Exception as e:
    print("❌ ChromaDB Initialization Failed:", e)
    collection = None

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logging.warning("Groq not installed. LLM features disabled.")

# CONFIG
load_dotenv()
GROQ_PRIMARY_MODEL = "llama-3.1-8b-instant"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ================= SEMANTIC MODEL FOR EXPLAINABLE AI =================

try:
    if SENTENCE_AVAILABLE:
        semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
        SEMANTIC_AVAILABLE = True
        logging.info("Semantic similarity model loaded")
    else:
        semantic_model = None
        SEMANTIC_AVAILABLE = False
except Exception as e:
    semantic_model = None
    SEMANTIC_AVAILABLE = False
    logging.warning(f"Semantic model failed: {e}")

# DATABASES
# Initialize with None first
collection = None

conn = sqlite3.connect("enterprise_ai.db", check_same_thread=False)
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS history (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    query TEXT,
    response TEXT,
    timestamp TEXT
)
""")
conn.commit()

# GROQ SERVICE
class GroqService:
    def __init__(self):
        self.client = None
        if GROQ_AVAILABLE:
            api_key = os.getenv("GROQ_API_KEY")
            if api_key:
                try:
                    self.client = Groq(api_key=api_key)
                except Exception as e:
                    logging.error(f"Groq initialization failed: {e}")
            else:
                logging.warning("GROQ_API_KEY not found in .env file")
        else:
            logging.info("Groq not available. Using mock responses.")

    def generate(self, prompt, system_role="You are an expert enterprise analyst.", temperature=0.3):
        if not self.client:
            # Mock response for testing
            return f"""Mock LLM Response to: {prompt[:100]}...
            
System Role: {system_role}
Temperature: {temperature}

This is a mock response because Groq API is not configured. Please:
1. Set GROQ_API_KEY in .env file
2. Install groq package: pip install groq"""
        
        try:
            res = self.client.chat.completions.create(
                model=GROQ_PRIMARY_MODEL,
                messages=[
                    {"role": "system", "content": system_role},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=2048
            )
            return res.choices[0].message.content
        except Exception as e:
            logging.error(f"Groq API error: {e}")
            return f"Error generating response: {str(e)}"

groq = GroqService()

# DOCUMENT PROCESSING (FIXED)
uploaded_texts = []
uploaded_files = []  # Track uploaded file names

# ================= EVALUATION METRICS =================
evaluation_metrics = {
    "query_times": [],
    "processing_times": [],
    "retrieval_precision": [],
    "decision_predictions": [],
    "decision_ground_truth": []
}

class DocumentProcessor:
    @staticmethod
    def extract_text(file):
        try:
            # Handle Gradio file object properly
            file_path = ""
            
            # Check file type and get path
            if isinstance(file, dict):
                # Newer Gradio versions pass dict
                file_path = file.get('name', '')
            elif hasattr(file, 'name'):
                # File-like object
                file_path = file.name
            elif isinstance(file, str):
                # String path
                file_path = file
            else:
                # Try to get as string
                file_path = str(file)
            
            if not file_path or not os.path.exists(file_path):
                raise ValueError(f"File not found or invalid: {file_path}")
            
            logging.info(f"Extracting text from: {file_path}")
            
            # Get file extension
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.pdf':
                with pdfplumber.open(file_path) as pdf:
                    text_parts = []
                    for i, page in enumerate(pdf.pages):
                        try:
                            text = page.extract_text()
                            if text and text.strip():
                                text_parts.append(text.strip())
                        except Exception as e:
                            logging.warning(f"Page {i+1} extraction failed: {e}")
                            continue
                    return "\n\n".join(text_parts)
                    
            elif file_ext in ['.docx', '.doc']:
                doc = docx.Document(file_path)
                text_parts = []
                for para in doc.paragraphs:
                    if para.text.strip():
                        text_parts.append(para.text.strip())
                return "\n\n".join(text_parts)
                
            else:
                # Try multiple encodings for text files
                encodings = ['utf-8', 'latin-1', 'cp1252', 'utf-16']
                for encoding in encodings:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            content = f.read()
                            if content.strip():
                                return content.strip()
                    except UnicodeDecodeError:
                        continue
                    except Exception:
                        continue
                
                # Final fallback
                with open(file_path, 'rb') as f:
                    content = f.read()
                    return content.decode('utf-8', errors='ignore').strip()
                        
        except Exception as e:
            error_msg = f"Error extracting text from {os.path.basename(str(file))}: {str(e)}"
            logging.error(error_msg)
            logging.error(traceback.format_exc())
            raise RuntimeError(error_msg)

    @staticmethod
    def chunk_text(text, size=500, overlap=100):
        if not text:
            return []
        
        words = text.split()
        if len(words) <= size:
            return [text]
        
        chunks = []
        for i in range(0, len(words), size - overlap):
            chunk = words[i:i + size]
            if chunk:
                chunks.append(" ".join(chunk))
        
        return chunks

    @staticmethod
    def eda_analysis(texts):
        if not texts:
            return {"Status": "No documents available", "Message": "Please upload documents first"}
            
        try:
            lengths = [len(t.split()) for t in texts]
            
            # Simple word frequency analysis
            all_text = " ".join(texts).lower()
            # Remove common punctuation and split
            import re
            words = re.findall(r'\b[a-z]{3,}\b', all_text)
            word_counts = Counter(words)
            
            return {
                "Status": "Success",
                "Document Count": len(texts),
                "Total Words": sum(lengths),
                "Average Words per Doc": int(np.mean(lengths)) if lengths else 0,
                "Min Words": min(lengths) if lengths else 0,
                "Max Words": max(lengths) if lengths else 0,
                "Top 10 Words": dict(word_counts.most_common(10)),
                "Total Unique Words": len(word_counts)
            }
        except Exception as e:
            return {"Status": "Error", "Message": f"EDA failed: {str(e)}"}

# KNOWLEDGE GRAPH + TOPICS (Lightweight versions)
class KnowledgeGraph:
    def __init__(self):
        self.counter = Counter()
        self.available = SPACY_AVAILABLE

    def build(self, texts):
        if not self.available or not texts:
            return
            
        try:
            for t in texts[:5]:  # Limit to first 5 texts for performance
                if len(t) > 10000:  # Limit text length
                    t = t[:10000]
                doc = nlp(t)
                entities = [e.text for e in doc.ents if len(e.text) > 1]
                self.counter.update(entities)
        except Exception as e:
            logging.error(f"Knowledge Graph build failed: {e}")
            self.available = False

    def stats(self):
        if not self.available:
            return {"Status": "spaCy not available", "Message": "Install: pip install spacy && python -m spacy download en_core_web_sm"}
        
        if not self.counter:
            return {"Status": "No entities found", "Message": "Upload documents and run analytics"}
        
        return {
            "Status": "Success",
            "Top Entities": dict(self.counter.most_common(10)),
            "Total Unique Entities": len(self.counter)
        }

kg = KnowledgeGraph()

# Initialize topic engine based on availability
topic_engine = None
if BERTOPIC_AVAILABLE:
    try:
        # Simple initialization without complex dependencies
        topic_engine = BERTopic(min_topic_size=2, verbose=False)
    except Exception as e:
        logging.error(f"BERTopic initialization failed: {e}")
        topic_engine = None
        BERTOPIC_AVAILABLE = False

# FEATURES
def compliance_check(text):
    try:
        if not text:
            return {"Status": "No text available", "Message": "Upload documents first"}
        
        rules = ["confidentiality", "data protection", "liability", "agreement", "terms", "conditions"]
        text_lower = text.lower()
        found = [r for r in rules if r in text_lower]
        missing = [r for r in rules if r not in text_lower]
        
        return {
            "Status": "PASS" if len(found) >= 3 else "WARNING" if found else "FAIL",
            "Found Clauses": found,
            "Missing Clauses": missing,
            "Score": f"{len(found)}/{len(rules)}"
        }
    except Exception as e:
        return {"Status": "Error", "Message": f"Compliance check failed: {str(e)}"}

def extract_dates(text):
    try:
        if not text:
            return []
        
        dates = []
        words = text.split()
        
        for i in range(len(words)):
            # Try single word
            word = words[i].strip('.,;:()[]{}"\'')

            # Skip obvious non-dates
            if len(word) < 3 or word.isnumeric() and len(word) != 4:
                continue
                
            try:
                parsed = date_parser.parse(word, fuzzy=False, default=datetime.now())
                date_str = str(parsed.date())
                if date_str not in dates:
                    dates.append(date_str)
            except:
                # Try two-word combinations
                if i < len(words) - 1:
                    phrase = f"{word} {words[i+1]}"
                    try:
                        parsed = date_parser.parse(phrase, fuzzy=True, default=datetime.now())
                        date_str = str(parsed.date())
                        if date_str not in dates:
                            dates.append(date_str)
                    except:
                        continue
        
        return sorted(set(dates))[:15]  # Limit to 15 dates
    except Exception as e:
        logging.error(f"Date extraction failed: {e}")
        return []

import threading

def run_background_analytics(texts):
    """Runs heavy ML tasks in a separate thread to avoid blocking the UI."""
    try:
        logging.info("Starting background analytics...")
        if texts:
            # Build knowledge graph
            kg.build(texts)
            
            # Topic modeling (only if available and not too much text)
            if BERTOPIC_AVAILABLE and topic_engine and len(texts) <= 10:
                try:
                    # Use smaller sample for topic modeling
                    sample_texts = texts[:10]
                    # Limit text length
                    sample_texts = [t[:5000] for t in sample_texts]
                    topic_engine.fit_transform(sample_texts)
                    logging.info("Topic modeling complete.")
                except Exception as e:
                    logging.error(f"Topic modeling failed: {e}")
            
            logging.info("Background analytics complete.")
        else:
            logging.warning("No texts for analytics")
    except Exception as e:
        logging.error(f"Background analytics failed: {e}")

def process_upload(files):
    logging.info(f"Processing upload: {files}")
    uploaded_texts.clear()
    uploaded_files.clear()
    
    # Defensive check
    if not files:
        return "No files uploaded."
    
    try:
        results = []
        success_count = 0
        total_chunks = 0
        
        for f in files:
            if f is None:
                continue
                
            try:
                # Get file name
                if isinstance(f, dict):
                    fname = f.get('name', 'unknown')
                elif hasattr(f, 'name'):
                    fname = f.name
                else:
                    fname = str(f)
                
                base_name = os.path.basename(fname)
                uploaded_files.append(base_name)
                
                # Extract text
                text = DocumentProcessor.extract_text(f)
                
                if not text or not text.strip():
                    results.append(f"⚠️ {base_name}: Empty or no text")
                    continue
                
                # Store text
                uploaded_texts.append(text)
                
                # Chunking
                chunks = DocumentProcessor.chunk_text(text)
                total_chunks += len(chunks)
                
                # Indexing (if ChromaDB is available)
                if collection and chunks:
                    try:
                        ids = [str(uuid.uuid4()) for _ in chunks]
                        metadatas = [{"source": base_name, "chunk": i} for i in range(len(chunks))]
                        collection.add(
                            documents=chunks,
                            ids=ids,
                            metadatas=metadatas
                        )
                        results.append(f"✅ {base_name}: {len(chunks)} chunks indexed")
                    except Exception as e:
                        logging.error(f"ChromaDB indexing failed: {e}")
                        results.append(f"✅ {base_name}: {len(chunks)} chunks processed (indexing skipped)")
                else:
                    results.append(f"✅ {base_name}: {len(chunks)} chunks processed")
                
                success_count += 1
                
            except Exception as inner_e:
                base_name = os.path.basename(str(getattr(f, 'name', 'unknown')))
                logging.error(f"Failed to process {base_name}: {inner_e}")
                results.append(f"❌ {base_name}: Failed - {str(inner_e)[:100]}")
        
        # Start background analytics if we have successful uploads
        if success_count > 0:
            thread = threading.Thread(target=run_background_analytics, args=(list(uploaded_texts),))
            thread.daemon = True
            thread.start()
            
            # Prepare summary
            summary = f"""
🎉 **Upload Complete!**

**Summary:**
• Successfully processed: {success_count} file(s)
• Total text chunks: {total_chunks}
• Files: {', '.join(uploaded_files[:3])}{'...' if len(uploaded_files) > 3 else ''}

**Next Steps:**
1. Go to **Ask AI (RAG)** tab to ask questions about your documents
2. Check **Analytics & Governance** for insights
3. Background analytics are running...

**Status:** ✅ Ready
            """
            return summary + "\n\n" + "\n".join(results)
        else:
            return "❌ No documents were successfully processed. Please check:\n• File formats (PDF, DOCX, TXT only)\n• File is not empty\n• File is not corrupted"
            
    except Exception as e:
        logging.error(f"Upload processing error: {e}")
        logging.error(traceback.format_exc())
        return f"❌ Critical error: {str(e)[:200]}"
    
def safe_summarize(summary_length):
    if not uploaded_texts:
        return "Please upload documents first."

    summaries = []
    max_chars = 2000  # Safe limit for Groq free tier

    for i, text in enumerate(uploaded_texts):
        chunk = text[:max_chars]

        prompt = f"""
Create a {summary_length.lower()} summary of the following document:

{chunk}
"""

        try:
            summary = groq.generate(prompt, temperature=0.2)
            summaries.append(f"### Document {i+1} Summary\n{summary}")
        except Exception as e:
            summaries.append(f"❌ Document {i+1} failed: {str(e)}")

    return "\n\n".join(summaries)

def get_safe_context(max_chars=1800):
    if not uploaded_texts:
        return ""
    # Use only first document + cap length
    return uploaded_texts[0][:max_chars]

# ================= EXPLAINABLE DECISION SUPPORT =================
def explainable_decision_support(query):
    if not query:
        return "Please enter a decision scenario."

    if not uploaded_texts:
        return "Please upload documents first."

    try:
        context = get_safe_context()

        # LLM reasoning
        prompt = f"""
You are an enterprise AI decision analyst.

Analyze the project and provide:

1. Decision (APPROVE / REJECT / REVIEW)
2. Risk Score (0 to 1)
3. Confidence Percentage
4. Key Reasons

Project Scenario:
{query}

Document Context:
{context}

Respond in structured format.
"""

        llm_response = groq.generate(prompt, temperature=0.2)

        # ================= IMPROVED DYNAMIC SCORING =================
        
        text_lower = context.lower()
        words = text_lower.split()
        total_words = max(len(words), 1)

        # Risk related terms
        risk_keywords = [
            "risk","delay","penalty","loss","uncertain","failure",
            "cost overrun","legal issue","conflict","problem","challenge"
        ]

        # Positive / compliance terms
        compliance_keywords = [
            "agreement","approved","valid","compliance",
            "feasible","secured","guarantee","confirmed"
        ]

        # Count frequency (not just presence)
        risk_count = sum(text_lower.count(w) for w in risk_keywords)
        compliance_count = sum(text_lower.count(w) for w in compliance_keywords)

        # Normalize by document size
        risk_density = risk_count / total_words
        compliance_density = compliance_count / total_words

        # Dynamic risk score (0–1)
        risk_score = min((risk_density * 10) - (compliance_density * 5), 1)
        risk_score = max(risk_score, 0)

        # Confidence based on text richness + risk
        confidence = int((1 - risk_score) * 100)

        # Decision logic
        if risk_score < 0.25:
            decision = "APPROVE"
        elif risk_score > 0.6:
            decision = "REJECT"
        else:
            decision = "REVIEW"

        # Log prediction for evaluation
        try:
            evaluation_metrics["decision_predictions"].append(decision)
        except Exception:
            pass

        return f"""
## ✅ Explainable Decision Report

### Decision: **{decision}**

### Risk Score: **{round(risk_score,2)}**

### Confidence: **{confidence}%**

### Key Factors:
- Compliance Indicators: {compliance_count}
- Risk Indicators: {risk_count}

### AI Reasoning:
{llm_response}
"""

    except Exception as e:
        return f"Decision analysis failed: {str(e)}"

# ================= SEMANTIC EXPLAINABLE DECISION SUPPORT =================
def semantic_explainable_decision(query):
    if not uploaded_texts:
        return "Please upload documents first."

    if not SEMANTIC_AVAILABLE:
        return "Semantic model not available. Install sentence-transformers."

    context = " ".join(uploaded_texts[:2])[:3000]

    # Risk concepts
    risk_concepts = [
        "financial risk",
        "legal issues",
        "project delay",
        "budget overrun",
        "technical failure",
        "compliance violation",
        "uncertainty in timeline",
        "security risk"
    ]

    # Positive concepts
    positive_concepts = [
        "project feasibility",
        "business alignment",
        "cost efficiency",
        "compliance satisfied",
        "low risk implementation",
        "successful delivery",
        "high performance outcome",
        "reliable system"
    ]

    # Encode document
    doc_embedding = semantic_model.encode(context, convert_to_tensor=True)

    # Encode concepts
    risk_embeddings = semantic_model.encode(risk_concepts, convert_to_tensor=True)
    positive_embeddings = semantic_model.encode(positive_concepts, convert_to_tensor=True)

    # Compute semantic similarity
    risk_scores = util.cos_sim(doc_embedding, risk_embeddings)[0]
    positive_scores = util.cos_sim(doc_embedding, positive_embeddings)[0]

    avg_risk = float(risk_scores.mean())
    avg_positive = float(positive_scores.mean())

    # Dynamic scoring
    risk_score = max(avg_risk - avg_positive, 0)
    risk_score = min(risk_score, 1)

    confidence = int((1 - risk_score) * 100)

    if risk_score < 0.25:
        decision = "APPROVE"
    elif risk_score > 0.6:
        decision = "REJECT"
    else:
        decision = "REVIEW"

    # Ask LLM for reasoning
    reasoning_prompt = f"""
Analyze the project document and explain the decision.

Decision: {decision}
Risk Score: {risk_score:.2f}

Document:
{context[:2000]}
"""

    reasoning = groq.generate(reasoning_prompt, temperature=0.2)

    # Log prediction for evaluation
    try:
        evaluation_metrics["decision_predictions"].append(decision)
    except Exception:
        pass

    return f"""
✅ **Semantic Explainable Decision Report**

Decision: {decision}
Risk Score: {round(risk_score,2)}
Confidence: {confidence}%

Semantic Risk Similarity: {round(avg_risk,3)}
Semantic Positive Similarity: {round(avg_positive,3)}

AI Reasoning:
{reasoning}
"""

def safe_insights(insight_type):
    if not uploaded_texts:
        return "Please upload documents first."

    context = get_safe_context()

    prompt = f"""
Provide {insight_type.lower()} based on the following document content:

{context}
"""
    return groq.generate(prompt, temperature=0.2)



def simple_text_search(query, texts, top_k=3):
    """Simple text search fallback when ChromaDB is not available"""
    if not texts:
        return []
    
    # Simple keyword matching
    query_words = set(query.lower().split())
    results = []
    
    for i, text in enumerate(texts):
        text_lower = text.lower()
        score = sum(1 for word in query_words if word in text_lower)
        if score > 0:
            # Get relevant snippet
            words = text.split()
            snippet = " ".join(words[:100]) + ("..." if len(words) > 100 else "")
            results.append((score, snippet))
    
    # Sort by score and return top_k
    results.sort(key=lambda x: x[0], reverse=True)
    return [r[1] for r in results[:top_k]]

# ================= DOCUMENT SIMILARITY =================
def compute_document_similarity():
    """
    Computes cosine similarity between uploaded documents
    """
    if len(uploaded_texts) < 2:
        return {
            "Status": "Need at least 2 documents",
            "Message": "Upload multiple documents"
        }

    try:
        # Convert documents to vectors
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(uploaded_texts)

        # Compute similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)

        results = []
        n = len(uploaded_texts)

        for i in range(n):
            for j in range(i + 1, n):
                score = float(similarity_matrix[i][j])

                relation = "Low"
                if score > 0.8:
                    relation = "Duplicate / Highly Similar"
                elif score > 0.5:
                    relation = "Related"
                elif score > 0.3:
                    relation = "Slightly Related"

                results.append({
                    "Doc 1": uploaded_files[i] if i < len(uploaded_files) else f"Doc {i+1}",
                    "Doc 2": uploaded_files[j] if j < len(uploaded_files) else f"Doc {j+1}",
                    "Similarity Score": round(score, 3),
                    "Relation": relation
                })

        return {
            "Status": "Success",
            "Total Comparisons": len(results),
            "Results": results
        }

    except Exception as e:
        return {"Status": "Error", "Message": str(e)}

def compute_retrieval_precision(query, retrieved_docs):
    """
    Simple precision metric:
    checks keyword overlap between query and retrieved text.
    """
    if not retrieved_docs:
        return 0

    query_words = set(query.lower().split())
    relevant_count = 0

    for doc in retrieved_docs:
        if any(word in doc.lower() for word in query_words):
            relevant_count += 1

    precision = relevant_count / len(retrieved_docs)
    evaluation_metrics["retrieval_precision"].append(precision)

    return precision


def rag_qa(user_id, query):
    if not query or not query.strip():
        return "Please enter a question."

    if not uploaded_texts:
        return "No documents uploaded yet. Please upload documents first in the **Document Ingestion** tab."

    start_time = time.time()

    # Get context
    context = ""
    retrieved_docs = []

    if collection:
        # Use ChromaDB vector search
        try:
            res = collection.query(
                query_texts=[query],
                n_results=3
            )
            if res.get("documents") and res["documents"][0]:
                retrieved_docs = res["documents"][0]
                context = "\n".join(retrieved_docs)
        except Exception as e:
            logging.error(f"ChromaDB query failed: {e}")
            # Fallback to simple search
            snippets = simple_text_search(query, uploaded_texts)
            retrieved_docs = snippets
            context = "\n".join(snippets)
    else:
        # Use simple text search
        snippets = simple_text_search(query, uploaded_texts)
        retrieved_docs = snippets
        context = "\n".join(snippets)

    if not context:
        context = "\n".join([t[:500] for t in uploaded_texts[:2]])  # Fallback to first documents
        retrieved_docs = [t[:500] for t in uploaded_texts[:2]]

    # Compute retrieval precision
    try:
        precision = compute_retrieval_precision(query, retrieved_docs)
    except Exception:
        precision = None

    # Generate answer
    prompt = f"""Based on the following document context, answer the question.

Context from documents:
{context[:3000]}

Question: {query}

Answer:"""

    answer = groq.generate(prompt, temperature=0.1)

    # Save to history
    try:
        cur.execute(
            "INSERT INTO history VALUES (?,?,?,?,?)",
            (str(uuid.uuid4()), user_id, query, answer, str(datetime.now()))
        )
        conn.commit()
    except Exception as e:
        logging.error(f"History save failed: {e}")

    end_time = time.time()
    evaluation_metrics["query_times"].append(end_time - start_time)

    return answer

def get_history():
    try:
        df = pd.read_sql("SELECT timestamp, query, response FROM history ORDER BY timestamp DESC LIMIT 50", conn)
        # Format timestamp
        if not df.empty and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        return df
    except Exception as e:
        logging.error(f"History load failed: {e}")
        return pd.DataFrame(columns=["timestamp", "query", "response"])

def get_topics():
    if not BERTOPIC_AVAILABLE or not topic_engine:
        return pd.DataFrame({
            "Topic": ["N/A"],
            "Name": ["Topic modeling not available"],
            "Count": [0]
        })
    
    try:
        topic_info = topic_engine.get_topic_info()
        if topic_info is not None and not topic_info.empty:
            return topic_info.head(10)
        else:
            return pd.DataFrame({
                "Topic": ["N/A"],
                "Name": ["No topics generated yet"],
                "Count": [0]
            })
    except Exception as e:
        logging.error(f"Topic retrieval failed: {e}")
        return pd.DataFrame({
            "Topic": ["Error"],
            "Name": [f"Failed to get topics: {str(e)[:100]}"],
            "Count": [0]
        })

def get_evaluation_report():
    try:
        avg_query_time = (
            sum(evaluation_metrics["query_times"]) / len(evaluation_metrics["query_times"])
            if evaluation_metrics["query_times"] else 0
        )

        avg_precision = (
            sum(evaluation_metrics["retrieval_precision"]) / len(evaluation_metrics["retrieval_precision"])
            if evaluation_metrics["retrieval_precision"] else 0
        )

        decision_acc = None
        if evaluation_metrics["decision_ground_truth"]:
            decision_acc = accuracy_score(
                evaluation_metrics["decision_ground_truth"],
                evaluation_metrics["decision_predictions"]
            )

        return {
            "Avg Query Latency (s)": round(avg_query_time, 3),
            "Avg Retrieval Precision": round(avg_precision, 3),
            "Decision Accuracy": decision_acc,
            "Total Queries": len(evaluation_metrics["query_times"])
        }
    except Exception as e:
        return {"Error": str(e)}

# ================= FEATURE 1: Show Stored Documents =================
def get_uploaded_docs():
    """Show uploaded documents"""
    if not uploaded_files:
        return {
            "Status": "No documents uploaded",
            "Files": []
        }

    return {
        "Status": "Success",
        "Uploaded Files": uploaded_files,
        "Total Files": len(uploaded_files),
        "Total Texts Stored": len(uploaded_texts)
    }

# ================= FEATURE 2: Show Vector DB Stats =================
def get_vector_db_stats():
    """Show vector database statistics"""
    if not collection:
        return {"Status": "ChromaDB not available"}

    try:
        count = collection.count()
        return {
            "Status": "Active",
            "Collection Name": "enterprise_docs",
            "Total Stored Chunks": count
        }
    except Exception as e:
        return {"Status": "Error", "Message": str(e)}

# ================= FEATURE 3: Show Document Metadata =================
def get_document_metadata():
    """Retrieve metadata from ChromaDB"""
    if not collection:
        return {"Status": "ChromaDB not available"}

    try:
        data = collection.get(limit=5)

        return {
            "Sample Documents": data.get("documents", [])[:3],
            "Metadata": data.get("metadatas", [])[:3],
            "IDs": data.get("ids", [])[:3]
        }
    except Exception as e:
        return {"Error": str(e)}

# ================= FEATURE 4: Delete / Reset Documents =================
def clear_database():
    """Delete all documents from vector DB"""
    global collection, chroma_client

    if not collection:
        return "ChromaDB not available"

    try:
        chroma_client.delete_collection("enterprise_docs")

        # recreate collection
        collection = chroma_client.get_or_create_collection(
            name="enterprise_docs",
            embedding_function=embedding_fn
        )

        uploaded_texts.clear()
        uploaded_files.clear()

        return "✅ Database cleared successfully"
    except Exception as e:
        return f"Error: {str(e)}"

# Enterprise CSS
custom_css = """
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.sidebar-content {
    background: #f8fafc !important;
}
#header {
    background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
    padding: 1.5rem;
    color: white;
    margin-bottom: 2rem;
    border-radius: 8px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    text-align: center;
}
/* ✅ ADD THIS PART */
.gradio-dataframe {
    max-height: 400px;
    overflow-y: auto;
}
.header-text p, .header-text h1 {
    color: white !important;
    text-align: center;
    font-weight: 700 !important;
    font-size: 2rem !important;
    margin: 0 !important;
}
.tabs {
    border-bottom: 2px solid #e2e8f0; 
}
.tab-nav {
    border: none !important;
    font-weight: 600;
}
.tab-nav.selected {
    border-bottom: 3px solid #2563eb !important;
    color: #1e3a8a !important;
    background: transparent !important;
}
.card {
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.5rem;
    background: white;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    transition: all 0.2s;
    margin-bottom: 1rem;
}
.card:hover {
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}
.upload-area {
    min-height: 200px;
}
.success-message {
    color: #059669;
    font-weight: 600;
}
.error-message {
    color: #dc2626;
    font-weight: 600;
}
.info-message {
    color: #2563eb;
    font-weight: 600;
}
button {
    font-weight: 600 !important;
}
"""

theme = gr.themes.Soft(
    primary_hue="blue",
    neutral_hue="slate",
    spacing_size="md",
    radius_size="lg"
).set(
    body_background_fill="#f8fafc",
    block_background_fill="white",
    block_border_width="1px",
    block_info_text_color="#64748b",
    block_label_text_size="sm",
    block_label_text_weight="600"
)

with gr.Blocks(title="AI Document Intelligence Platform", theme=theme, css=custom_css) as demo:
    with gr.Column(elem_id="header"):
        gr.Markdown("# AI Document Intelligence Platform", elem_classes=["header-text"])
        gr.Markdown("Upload, analyze, and query your documents with AI", elem_classes=["header-text"])
    
    user_id = gr.State(str(uuid.uuid4()))

    with gr.Tabs():
        # ================= DOCUMENT INGESTION =================
        with gr.TabItem("📄 Document Ingestion"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Upload Documents")
                    files = gr.File(
                        file_count="multiple", 
                        label="Upload Documents (PDF, DOCX, TXT)",
                        file_types=[".pdf", ".docx", ".txt"],
                        elem_classes=["card", "upload-area"]
                    )
                    gr.Markdown("""
                    **Supported formats:**
                    - PDF documents
                    - Word documents (.docx)
                    - Text files (.txt)
                    
                    **Note:** Large documents may take longer to process.
                    """)
                with gr.Column(scale=2):
                    gr.Markdown("### System Status")
                    out = gr.Textbox(
                        label="Processing Status", 
                        interactive=False, 
                        elem_classes=["card"],
                        lines=15
                    )
                    gr.Markdown("""
                    **Expected output:**
                    - ✅ File processed successfully
                    - ❌ File failed to process
                    - ⚠️ File skipped (empty or invalid)
                    """)
            
            files.upload(
                process_upload,
                inputs=files,
                outputs=out,
                queue=True
            )

        # ================= SUMMARIZATION =================
        with gr.TabItem("📝 Summarization"):
            with gr.Column(elem_classes=["card"]):
                gr.Markdown("### Document Summarization")
                gr.Markdown("Generate concise summaries of your uploaded documents.")
                
                with gr.Row():
                    summary_length = gr.Radio(
                        ["Short (1-2 paragraphs)", "Medium (3-4 paragraphs)", "Detailed (5+ paragraphs)"],
                        label="Summary Length",
                        value="Medium (3-4 paragraphs)"
                    )
                
                summary_output = gr.Markdown(label="Summary", elem_classes=["card"])
                
                gr.Button("✨ Generate Summary", variant="primary", size="lg").click(
    safe_summarize,
    inputs=summary_length,
    outputs=summary_output
)


        # ================= ASK AI (RAG) =================
        with gr.TabItem("🤖 Ask AI (RAG)"):
            with gr.Column(elem_classes=["card"]):
                gr.Markdown("### Ask Questions About Your Documents")
                gr.Markdown("Ask any question based on your uploaded documents.")
                
                q = gr.Textbox(
                    label="Your Question",
                    placeholder="E.g., What are the main points in the document? What dates are mentioned?",
                    lines=2
                )
                
                a = gr.Markdown(
                    label="AI Answer",
                    elem_classes=["card"]
                )
                
                with gr.Row():
                    ask_btn = gr.Button("🔍 Ask AI", variant="primary", size="lg")
                    clear_btn = gr.Button("Clear", variant="secondary")
                
                ask_btn.click(
                    rag_qa,
                    inputs=[user_id, q],
                    outputs=a
                )
                
                clear_btn.click(
                    lambda: ("", ""),
                    outputs=[q, a]
                )

        # ================= DECISION SUPPORT =================
        with gr.TabItem("⚖️ Decision Support"):
            with gr.Column(elem_classes=["card"]):
                gr.Markdown("### Decision Support System")
                gr.Markdown("Get AI-powered decision analysis based on your documents.")
                
                decision_query = gr.Textbox(
                    label="Decision Scenario",
                    placeholder="E.g., Should we approve this contract? What are the risks in this agreement?",
                    lines=3
                )
                
                decision_output = gr.Markdown(
                    label="Decision Analysis",
                    elem_classes=["card"]
                )
                
                gr.Button("📊 Analyze Decision", variant="primary", size="lg").click(
                    semantic_explainable_decision,
                    inputs=decision_query,
                    outputs=decision_output
                )

        # ================= DEEP INSIGHTS =================
        with gr.TabItem("🔍 Deep Insights"):
            with gr.Column(elem_classes=["card"]):
                gr.Markdown("### Deep Document Insights")
                gr.Markdown("Get comprehensive analysis and insights from your documents.")
                
                insights_type = gr.Radio(
                    ["Key Findings", "Themes & Patterns", "Action Items", "Risk Analysis"],
                    label="Insight Type",
                    value="Key Findings"
                )
                
                insights_output = gr.Markdown(
                    label="Generated Insights",
                    elem_classes=["card"]
                )
                
                gr.Button("✨ Generate Insights", variant="primary", size="lg").click(
                    lambda insight_type: groq.generate(f"Provide {insight_type.lower()} from these documents: {' '.join(uploaded_texts[:3000])}") if uploaded_texts else "Please upload documents first.",
                    inputs=insights_type,
                    outputs=insights_output
                )

        # ================= ANALYTICS & GOVERNANCE =================
        with gr.TabItem("📊 Analytics & Governance"):
            with gr.Tabs():
                # EDA Dashboard
                with gr.TabItem("📈 EDA Dashboard"):
                    with gr.Column(elem_classes=["card"]):
                        gr.Markdown("### Exploratory Data Analysis")
                        gr.Markdown("Basic statistics and analysis of your documents.")
                        eda_output = gr.JSON(label="EDA Results", elem_classes=["card"])
                        gr.Button("📊 Run EDA Analysis", variant="primary").click(
                            lambda: DocumentProcessor.eda_analysis(uploaded_texts),
                            outputs=eda_output
                        )

                # Knowledge Graph
                with gr.TabItem("🔗 Knowledge Graph"):
                    with gr.Column(elem_classes=["card"]):
                        gr.Markdown("### Entity Recognition")
                        gr.Markdown("Named entities extracted from your documents.")
                        kg_output = gr.JSON(label="Entity Distribution", elem_classes=["card"])
                        gr.Button("🔍 Extract Entities", variant="primary").click(
                            lambda: kg.stats(),
                            outputs=kg_output
                        )

                # Topic Modeling
                with gr.TabItem("🗂️ Topic Modeling"):
                    with gr.Column(elem_classes=["card"]):
                        gr.Markdown("### Topic Modeling")
                        gr.Markdown("Main topics discovered in your documents.")
                        topic_output = gr.Dataframe(
                            label="Identified Topics",
                            elem_classes=["card"],
                            interactive=False
                        )
                        gr.Button("📚 Identify Topics", variant="primary").click(
                            get_topics,
                            outputs=topic_output
                        )

                # Compliance Checker
                with gr.TabItem("✅ Compliance Checker"):
                    with gr.Column(elem_classes=["card"]):
                        gr.Markdown("### Compliance Analysis")
                        gr.Markdown("Check for important compliance clauses in documents.")
                        compliance_output = gr.JSON(label="Compliance Report", elem_classes=["card"])
                        gr.Button("⚖️ Run Compliance Check", variant="primary").click(
                            lambda: compliance_check(" ".join(uploaded_texts)) if uploaded_texts else {"Status": "No documents"},
                            outputs=compliance_output
                        )

                # Timeline Extraction
                with gr.TabItem("📅 Timeline Extraction"):
                    with gr.Column(elem_classes=["card"]):
                        gr.Markdown("### Date Extraction")
                        gr.Markdown("Important dates extracted from your documents.")
                        dates_output = gr.JSON(label="Key Dates", elem_classes=["card"])
                        gr.Button("📅 Extract Dates", variant="primary").click(
                            lambda: extract_dates(" ".join(uploaded_texts)) if uploaded_texts else [],
                            outputs=dates_output
                        )

                # User History
                with gr.TabItem("📋 User History"):
                    with gr.Column(elem_classes=["card"]):
                        gr.Markdown("### Interaction History")
                        gr.Markdown("Your previous questions and answers.")
                        history_output = gr.Dataframe(
                            label="Interaction Log",
                            elem_classes=["card"],
                            interactive=False,
                        )
                        gr.Button("🔄 Load History", variant="primary").click(
                            get_history,
                            outputs=history_output
                        )

                # Document Storage Viewer
                with gr.TabItem("📁 Stored Documents"):
                    with gr.Column(elem_classes=["card"]):
                        gr.Markdown("### Stored Documents")
                        docs_output = gr.JSON(label="Stored Files")
                        gr.Button("Show Stored Documents").click(
                            get_uploaded_docs,
                            outputs=docs_output
                        )

                # Vector Database Stats
                with gr.TabItem("🧠 Vector DB Stats"):
                    with gr.Column(elem_classes=["card"]):
                        gr.Markdown("### Vector Database Information")
                        db_output = gr.JSON(label="Database Stats")
                        gr.Button("Check DB Status").click(
                            get_vector_db_stats,
                            outputs=db_output
                        )

                # Metadata Viewer
                with gr.TabItem("📑 Metadata Viewer"):
                    with gr.Column(elem_classes=["card"]):
                        gr.Markdown("### Document Metadata")
                        meta_output = gr.JSON(label="Metadata")
                        gr.Button("Load Metadata").click(
                            get_document_metadata,
                            outputs=meta_output
                        )

                # Database Reset
                with gr.TabItem("🗑️ Reset Database"):
                    with gr.Column(elem_classes=["card"]):
                        gr.Markdown("### Clear Stored Documents")
                        reset_output = gr.Textbox(label="Status")
                        gr.Button("Delete All Documents", variant="stop").click(
                            clear_database,
                            outputs=reset_output
                        )

                # Document Similarity
                with gr.TabItem("📑 Document Similarity"):
                    with gr.Column(elem_classes=["card"]):
                        gr.Markdown("### Document Similarity Detection")
                        gr.Markdown("Detect duplicate or related documents using cosine similarity.")

                        similarity_output = gr.JSON(label="Similarity Results")

                        gr.Button("🔍 Check Similarity", variant="primary").click(
                            compute_document_similarity,
                            outputs=similarity_output
                        )

                # Evaluation Metrics
                with gr.TabItem("🧾 Evaluation Metrics"):
                    with gr.Column(elem_classes=["card"]):
                        gr.Markdown("### Evaluation Metrics")
                        eval_output = gr.JSON(label="Evaluation Metrics")
                        gr.Button("Show Evaluation Metrics").click(
                            get_evaluation_report,
                            outputs=eval_output
                        )

if __name__ == "__main__":
    # Display startup information
    print("=" * 60)
    print("AI Document Intelligence Platform")
    print("=" * 60)
    print("\nStarting server...")
    
    # Package status
    status = {
        "Gradio": "✓",
        "PDF Processing": "✓" if "pdfplumber" in globals() else "✗",
        "Word Processing": "✓" if "docx" in globals() else "✗",
        "ChromaDB": "✓" if CHROMADB_AVAILABLE else "✗ (using fallback)",
        "spaCy": "✓" if SPACY_AVAILABLE else "✗ (optional)",
        "BERTopic": "✓" if BERTOPIC_AVAILABLE else "✗ (optional)",
        "Groq": "✓" if GROQ_AVAILABLE else "✗ (mock mode)"
    }
    
    print("\nFeature Status:")
    for feature, stat in status.items():
        print(f"  {feature}: {stat}")
    
    if not GROQ_AVAILABLE:
        print("\n⚠️  Groq API is in mock mode. For full functionality:")
        print("  1. pip install groq")
        print("  2. Create .env file with GROQ_API_KEY=your_key")
    
    print("\n" + "=" * 60)
    
    # Launch with appropriate settings
    try:
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            debug=False
        )
    except Exception as e:
        print(f"\n❌ Failed to launch: {e}")
        print("\nTrying alternative port...")
        try:
            demo.launch(
                server_name="0.0.0.0",
                server_port=7861,
                share=False,
                show_error=True
            )
        except:
            print("❌ Could not start server. Check if port is already in use.")