import hashlib
import os
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

# Fix OpenMP conflict between different libraries (numpy, scipy, scikit-learn, pytorch)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import easyocr
import numpy as np
import ollama
import streamlit as st
from PIL import Image
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# khai báo thư viện neo4j cho greaph RAG
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_ollama import ChatOllama
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain

from langchain_core.prompts.prompt import PromptTemplate

# Định nghĩa lại Prompt cho Cypher để tránh dùng hàm lạ
CYPHER_GENERATION_TEMPLATE = """
Task: Generate a valid Cypher query to answer a user's question.
CRITICAL: You MUST use CONTAINS to search text, NOT exact property matching.

Instructions:
1. Dữ liệu trong database sử dụng TIẾNG VIỆT. Giữ nguyên khi tìm kiếm.
2. CHỈ sử dụng Cypher operators: CONTAINS, IN, STARTS WITH
   FORBIDDEN: LIKE, %, ILIKE, property: 'exact_value' (SQL syntax)
3. ALWAYS use WHERE clause with CONTAINS:
   ✅ WHERE n.id CONTAINS 'keyword' OR n.description CONTAINS 'keyword'
   ❌ MATCH (n:Project {description: 'project creation'})  (exact match, won't find)
4. Search on both 'id' and 'description' properties for better results:
   MATCH (n) WHERE n.id CONTAINS 'term' OR n.description CONTAINS 'term' RETURN n LIMIT 10
5. Always RETURN n with all properties.

Schema:
{schema}

Question: {question}
Cypher Query:"""
CYPHER_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], 
    template=CYPHER_GENERATION_TEMPLATE
)

# 2. Prompt để trả lời câu hỏi dựa trên kết quả truy vấn (Cần context và question)
QA_TEMPLATE = """
Bạn là một trợ lý thông minh cho hệ thống SmartDoc AI.
Dưới đây là dữ liệu từ đồ thị (Context gồm các nodes, properties, và relationships) và câu hỏi của người dùng.

Nhiệm vụ:
1. Phân tích context từ các nodes được trả về (description, id, relationships).
2. Nếu context có dữ liệu liên quan, hãy trả lời câu hỏi dựa trên các mối liên hệ và nội dung của nodes.
3. Trả lời bằng tiếng Việt, chi tiết và rõ ràng dựa trên dữ liệu từ graph.
4. Nếu Context TRỐNG hoặc KHÔNG liên quan đến câu hỏi, hãy trả lời:
"Xin lỗi, đồ thị dữ liệu hiện tại không chứa thông tin để trả lời câu hỏi này."

Context (nodes và properties từ Neo4j):
{context}

Question: {question}
Answer:"""

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"], 
    template=QA_TEMPLATE
)

#Database cho graph RAG
graph = Neo4jGraph(
    url="neo4j+s://a954d07c.databases.neo4j.io",
    username="neo4j",
    password="BNIJzfLaH-c5QOQqNAyS9igjQ15Ufe1uECTXYrVukL0"
)


# Khởi tạo mô hình Chat (Graph Transformer yêu cầu mô hình Chat thay vì LLM thường)
chat_llm = ChatOllama(model="qwen2.5:1.5b", temperature=0.7, repeat_penalty=1.1)

# Khởi tạo công cụ biến đổi văn bản thành Graph
graph_transformer = LLMGraphTransformer(llm=chat_llm)


# Giả sử bạn đã có biến `chunks` từ PDFPlumber và TextSplitter ở code cũs
# Tối ưu lại hàm bóc tách để tránh bị timeout trên Streamlit
def process_text_to_graph(chunks, graph_db):
    try:
        st.write("🔄 Bắt đầu xử lý dữ liệu đồ thị...")
        
        # Xóa dữ liệu cũ
        graph_db.query("MATCH (n) DETACH DELETE n")
        st.write("✅ Đã xóa dữ liệu cũ")
        
        # 1. Trích xuất tài liệu đồ thị
        with st.spinner("📊 Trích xuất tài liệu đồ thị từ chunks..."):
            doc = graph_transformer.convert_to_graph_documents(chunks)
            st.write(f"✅ Trích xuất được {len(doc)} graph documents")
        
        # 2. Đẩy vào DB
        if doc:
            with st.spinner("💾 Đẩy dữ liệu vào Neo4j..."):
                graph_db.add_graph_documents(
                    doc, 
                    include_source=True,
                )
            st.write(f"✅ Đẩy thành công {len(doc)} documents")
            
            # 3. Thêm nội dung context vào neo4j (để trả lời có nội dung)
            st.write("📝 Thêm nội dung context...")
            context_added = 0
            for idx, graph_doc in enumerate(doc):
                try:
                    if graph_doc.source:
                        # Lấy nội dung PDF chunk
                        full_content = graph_doc.source.page_content
                        
                        # Chia nội dung thành các phần nhỏ hơn (5KB mỗi phần) để dễ quản lý
                        chunk_size = 5000  # 5KB per context chunk
                        overlap = 500      # 500 chars overlap
                        
                        content_chunks = []
                        for i in range(0, len(full_content), chunk_size - overlap):
                            chunk = full_content[i:i + chunk_size]
                            if chunk.strip():  # Only if not empty
                                content_chunks.append(chunk)
                        
                        # Tạo CONTEXT nodes cho mỗi phần
                        for chunk_idx, content in enumerate(content_chunks):
                            try:
                                context_node_id = f"CONTEXT_{idx}_{chunk_idx}"
                                create_context = """
                                CREATE (ctx:CONTEXT {
                                    id: $ctx_id,
                                    content: $ctx_content,
                                    chunk_index: $chunk_idx
                                })
                                """
                                graph_db.query(create_context, {
                                    "ctx_id": context_node_id,
                                    "ctx_content": content,
                                    "chunk_idx": chunk_idx
                                })
                                context_added += 1
                                
                                # Link TẤT CẢ nodes (trừ CONTEXT) tới CONTEXT này
                                # Đảm bảo mọi node đều có context liên kết
                                link_context = """
                                MATCH (ctx:CONTEXT {id: $ctx_id})
                                MATCH (e)
                                WHERE NOT (e:CONTEXT)
                                CREATE (e)-[:HAS_CONTEXT]->(ctx)
                                """
                                try:
                                    graph_db.query(link_context, {"ctx_id": context_node_id})
                                except:
                                    pass  # If link fails, continue
                            except Exception as e:
                                pass  # Skip this chunk if fails
                except Exception as e:
                    pass  # Continue with other documents
            
            if context_added > 0:
                st.write(f"✅ Thêm {context_added} context chunks + liên kết entities")
            
            # 4. Kiểm tra dữ liệu
            try:
                result = graph_db.query("MATCH (n) RETURN count(n) as total_nodes")
                total_nodes = result[0]["total_nodes"] if result else 0
                st.write(f"📈 Tổng nodes trong Neo4j: {total_nodes}")
            except Exception as e:
                st.info(f"ℹ️ Không thể kiểm tra nodes: {str(e)[:60]}")
            
            return len(doc)
        
    except Exception as e:
        error_msg = str(e)
        st.error(f"❌ Lỗi xử lý graph: {error_msg[:200]}")
        
        # Debug thêm
        if "UNIQUE constraint" in error_msg:
            st.info("💡 Hint: Có thể dữ liệu bị duplicate, hãy xóa Neo4j và thử lại")
        elif "Connection refused" in error_msg:
            st.error("🚨 Không kết nối được Neo4j. Kiểm tra URL/username/password")
        
        return 0

def ask_graph_rag(question: str, graph_db, llm_model):
    """
    Improved Graph RAG - uses node properties for better answers
    """
    try:
        # Step 1: Extract meaningful keywords
        stopwords = {'làm', 'cách', 'có', 'là', 'được', 'gì', 'nào', 'nên', 'hãy', 'để', 'từ', 'và', 'hay', 'như', 'tại', 'với', 'trong', 'trên', 'dưới', 'về', 'sau', 'trước', 'này', 'kia', 'không', 'sao', 'ai', 'nước', 'người', 'điều', 'còn', 'việc', 'mà', 'nhất', 'thì'}
        keywords = [w.strip().lower() for w in question.split() if len(w.strip()) > 2 and w.strip().lower() not in stopwords]
        
        if not keywords:
            return "❌ Câu hỏi quá chung chung. Hãy dùng từ khóa cụ thể hơn (ví dụ: 'Django', 'AWS', 'project')."
        
        # Step 2: Build search query - use existing properties (id, description)
        where_clauses = []
        for kw in keywords[:3]:
            where_clauses.append(f"n.id CONTAINS '{kw}'")
            where_clauses.append(f"n.description CONTAINS '{kw}'")
        
        where_condition = " OR ".join(where_clauses)
        
        # Fetch data from existing node properties + follow relationship to context
        search_query = f"""
        MATCH (n) 
        WHERE {where_condition}
        OPTIONAL MATCH (n)-[:HAS_CONTEXT]->(ctx:CONTEXT)
        RETURN DISTINCT n.id as node_id, n.description as description, labels(n) as labels, 
               collect({{id: ctx.id, content: ctx.content, idx: ctx.chunk_index}}) as contexts
        LIMIT 6
        """
        
        st.write(f"🔍 Searching Graph for: {', '.join(keywords)}")
        results = graph_db.query(search_query)
        
        # FALLBACK: If no results found, get some popular nodes
        if not results:
            st.write(f"⚠️ Không tìm được từ khóa '{', '.join(keywords)}', đang tìm các entities nổi bật...")
            fallback_query = """
            MATCH (n)
            WHERE NOT (n:CONTEXT)
            OPTIONAL MATCH (n)-[:HAS_CONTEXT]->(ctx:CONTEXT)
            RETURN DISTINCT n.id as node_id, n.description as description, labels(n) as labels,
                   collect({id: ctx.id, content: ctx.content, idx: ctx.chunk_index}) as contexts
            LIMIT 6
            """
            results = graph_db.query(fallback_query)
        
        # Step 3: Filter and prepare context from entities
        # Priority: Keep entities that have EITHER description OR context
        valid_results = []
        
        # Get all available contexts as fallback
        all_contexts_query = "MATCH (ctx:CONTEXT) RETURN ctx.id as ctx_id, ctx.content as content, ctx.chunk_index as idx LIMIT 10"
        try:
            all_contexts = graph_db.query(all_contexts_query)
        except:
            all_contexts = []
        
        if not results:
            return f"❌ Không tìm được entities nào trong graph. Vui lòng upload PDF và xử lý lại."
        
        for r in results:
            description = r.get('description', '')
            contexts = r.get('contexts', []) or []  # List of context objects
            
            # Filter out empty contexts
            contexts = [c for c in contexts if c and c.get('content')]
            
            # FALLBACK: If entity has no context, assign some available contexts
            if not contexts and all_contexts:
                contexts = all_contexts[:2]  # Assign first 2 available contexts
            
            # Keep entity if it has description OR context (not too strict)
            if (description and len(str(description)) > 20) or contexts:
                valid_results.append({
                    'id': r.get('node_id', 'N/A'),
                    'text': description or "[No description]",
                    'labels': r.get('labels', []),
                    'contexts': contexts  # List of context chunks
                })
        
        if not valid_results:
            return f"❌ Không tìm được entities nào với context cho '{', '.join(keywords)}'."
        
        # Step 4: Format comprehensive context
        context_parts = []
        
        # Add entity nodes with their linked contexts
        for i, r in enumerate(valid_results[:3], 1):  # Top 3 entities
            node_id = r['id']
            text = r['text']
            labels = r['labels']
            contexts = r.get('contexts', []) or []  # List of context chunks
            
            label_str = ', '.join(labels) if labels else 'Entity'
            context_parts.append(f"[Entity {i}: {label_str}] {node_id}")
            
            # Show description if meaningful
            if text and text != "[No description]":
                text_clean = text.replace('\n\n', '\n').replace('\r', '').strip()
                if len(text_clean) > 400:
                    text_clean = text_clean[:400]
                    last_period = text_clean.rfind('.')
                    if last_period > 200:
                        text_clean = text_clean[:last_period+1]
                context_parts.append(f"Mô tả: {text_clean}")
            
            # PRIORITIZE: Show context content chunks (focused, not too long)
            if contexts:
                # Sort by chunk index if available
                contexts_sorted = sorted(contexts, key=lambda x: x.get('idx', 0))
                context_parts.append(f"📄 Nội dung liên quan ({len(contexts_sorted)} phần):")
                
                for ctx_idx, ctx in enumerate(contexts_sorted[:2], 1):  # Show max 2 chunks per entity
                    content = ctx.get('content', '')
                    if content and len(str(content)) > 50:
                        content_clean = str(content).replace('\n\n', '\n').strip()[:600]
                        context_parts.append(f"  Phần {ctx_idx}:\n{content_clean}")
            elif not text or text == "[No description]":
                context_parts.append("(Chưa có nội dung liên quan từ tài liệu)")
            
            context_parts.append("---")
        
        
        context_text = "\n".join(context_parts)
        context_count = sum(1 for r in valid_results if r.get('contexts'))
        total_chunks = sum(len(r.get('contexts', []) or []) for r in valid_results)
        
        # All entities should have context now
        if len(valid_results) == context_count and context_count > 0:
            entity_ids = ", ".join([r['id'] for r in valid_results])
            st.caption(f"✅ Tìm được {len(valid_results)} entities: {entity_ids}")
            st.caption(f"✅ TẤT CẢ đều có context ({total_chunks} chunks)")
        elif context_count > 0:
            st.caption(f"📊 Tìm được {len(valid_results)} entities, {context_count} có {total_chunks} content chunks")
        else:
            st.warning(f"⚠️ Tìm được {len(valid_results)} entities nhưng chưa có PDF content liên quan")
        
        # Step 5: Create powerful prompt for Graph-specific queries
        graph_prompt = f"""Bạn là chuyên gia phân tích tài liệu dự án.
Dựa trên TOÀN BỘ thông tin từ tài liệu dưới đây, hãy trả lời câu hỏi một cách CHI TIẾT và CHÍNH XÁC.

=== THÔNG TIN ĐẦY ĐỦ TỪ TÀI LIỆU ===
{context_text}

=== CÂU HỎI ===
{question}

=== HƯỚNG DẪN TRẢ LỜI ===
1. Sử dụng CHỈ thông tin từ tài liệu trên, không bịa chuyện
2. Trả lời CHI TIẾT, từng bước, với ví dụ cụ thể từ tài liệu
3. Nếu tài liệu đề cập nhiều bước/phần, hãy liệt kê hết
4. Nếu thiếu thông tin, nói rõ "Tài liệu không nêu"
5. Format: Dùng số thứ tự (1., 2., 3.) cho các bước

=== TRẢ LỜI CHI TIẾT ==="""
        
        with st.spinner("🤖 Graph RAG đang xử lý..."):
            model_name = llm_model.model if hasattr(llm_model, 'model') else 'qwen2.5:1.5b'
            response = generate_with_ollama(model_name, graph_prompt)
        
        return response
    
    except Exception as e:
        error_msg = str(e)
        st.error(f"⚠️ Graph search error: {error_msg[:150]}")
        return f"❌ Lỗi: {error_msg[:100]}"


st.set_page_config(page_title="SmartDoc AI", page_icon="📄", layout="wide")


def init_state() -> None:
    defaults = {
        "retriever": None,
        "vector_store": None,
        "chat_history": [],
        "active_file": None,
        "image_context": None,
        "image_file_key": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource(show_spinner=False)
def get_ollama_client() -> ollama.Client:
    host = os.getenv("OLLAMA_HOST")
    if host:
        return ollama.Client(host=host)
    return ollama.Client()


class LocalOllamaEmbeddings(Embeddings):
    def __init__(self, model: str, client: ollama.Client) -> None:
        self.model = model
        self.client = client

    def _extract_embeddings(self, response) -> List[List[float]]:
        if hasattr(response, "embeddings"):
            return response.embeddings
        if isinstance(response, dict) and "embeddings" in response:
            return response["embeddings"]
        raise ValueError("Unexpected embedding response format from Ollama.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        response = self.client.embed(model=self.model, input=texts)
        return self._extract_embeddings(response)

    def embed_query(self, text: str) -> List[float]:
        embeddings = self.embed_documents([text])
        return embeddings[0]


@st.cache_resource(show_spinner=False)
def get_embedder() -> LocalOllamaEmbeddings:
    return LocalOllamaEmbeddings(model="nomic-embed-text", client=get_ollama_client())


def generate_with_ollama(model_name: str, prompt_text: str) -> str:
    response = get_ollama_client().generate(
        model=model_name,
        prompt=prompt_text,
        options={
            "temperature": 0.7,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
        },
    )

    if hasattr(response, "response"):
        return response.response
    if isinstance(response, dict) and "response" in response:
        return response["response"]
    raise ValueError("Unexpected generation response format from Ollama.")


def is_vietnamese(text: str) -> bool:
    vietnamese_chars = "aadeioouuyáàảãạăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ"
    lowered = text.lower()
    return any(char in lowered for char in vietnamese_chars)


def build_prompt(is_vi: bool) -> PromptTemplate:
    if is_vi:
        template = (
            "Su dung ngu canh sau day de tra loi cau hoi. "
            "Neu khong co du lieu, hay noi rang ban khong biet. "
            "Tra loi ngan gon (3-4 cau) bang tieng Viet.\n\n"
            "Ngu canh:\n{context}\n\n"
            "Cau hoi: {question}\n\n"
            "Tra loi:"
        )
    else:
        template = (
            "Use the context below to answer the question. "
            "If the context is insufficient, say you don't know. "
            "Keep the answer concise in 3-4 sentences.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )

    return PromptTemplate(template=template, input_variables=["context", "question"])


def build_retriever(pdf_path: str, chunk_size: int, chunk_overlap: int, top_k: int):
    loader = PDFPlumberLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)

    embedder = get_embedder()
    vector_store = FAISS.from_documents(chunks, embedder)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )
    return retriever, vector_store, len(docs), len(chunks), chunks

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".tif"}


def is_supported_image(filename: str) -> bool:
    return Path(filename).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


@st.cache_resource(show_spinner=False)
def get_easyocr_reader() -> easyocr.Reader:
    return easyocr.Reader(["en", "vi"], gpu=False)


def load_image_bytes(image_bytes: bytes) -> np.ndarray:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return np.array(image)


def extract_text_from_image_bytes(image_bytes: bytes) -> str:
    image_array = load_image_bytes(image_bytes)
    results = get_easyocr_reader().readtext(image_array, detail=0)
    return "\n".join(results).strip()


def compute_image_upload_key(files) -> str:
    hash_obj = hashlib.md5()
    for uploaded_file in sorted(files, key=lambda f: f.name):
        content = uploaded_file.getvalue()
        hash_obj.update(uploaded_file.name.encode("utf-8"))
        hash_obj.update(str(uploaded_file.size).encode("utf-8"))
        # include a small sample of bytes so identical names/sizes with different content are handled
        hash_obj.update(content[:1024])
    return hash_obj.hexdigest()


def gather_images_from_uploaded_files(files):
    images = []
    for uploaded_file in files:
        content = uploaded_file.getvalue()
        if uploaded_file.name.lower().endswith(".zip") or zipfile.is_zipfile(BytesIO(content)):
            with zipfile.ZipFile(BytesIO(content)) as archive:
                for info in archive.infolist():
                    if not info.is_dir() and is_supported_image(info.filename):
                        images.append((Path(info.filename).name, archive.read(info)))
        elif is_supported_image(uploaded_file.name):
            images.append((uploaded_file.name, content))
    return images


def ask_rag(question: str, retriever, model_name: str) -> Tuple[str, List[Dict[str, str]]]:
    prompt = build_prompt(is_vietnamese(question))
    source_docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in source_docs)
    formatted_prompt = prompt.format(context=context, question=question)
    answer = generate_with_ollama(model_name, formatted_prompt)

    sources: List[Dict[str, str]] = []
    for doc in source_docs:
        page = doc.metadata.get("page", "N/A")
        snippet = doc.page_content[:280].replace("\n", " ")
        sources.append({"page": str(page), "snippet": snippet})

    return answer, sources


def ask_image_question(question: str, image_context: str, model_name: str) -> str:
    prompt = build_prompt(is_vietnamese(question))
    formatted_prompt = prompt.format(context=image_context, question=question)
    return generate_with_ollama(model_name, formatted_prompt)


def render_sidebar() -> Tuple[int, int, int, str, bool]:
    with st.sidebar:
        st.title("SmartDoc AI")
        st.caption("RAG system for PDF Q&A")

        st.subheader("Settings")
        chunk_size = st.slider("Chunk size", min_value=500, max_value=2000, value=1000, step=100)
        chunk_overlap = st.slider("Chunk overlap", min_value=50, max_value=300, value=100, step=25)
        top_k = st.slider("Top-k retrieval", min_value=1, max_value=10, value=3, step=1)
        model_name = st.text_input("Ollama model", value="qwen2.5:1.5b")
        st.caption("Low-RAM recommendation: qwen2.5:1.5b or qwen2.5:3b")
        
        # Option to disable Graph RAG (if having Neo4j issues)
        enable_graph_rag = st.checkbox("Enable Graph RAG", value=True, help="Uncheck if Neo4j is having issues")

        st.subheader("Chat History")
        if st.session_state.chat_history:
            for idx, item in enumerate(reversed(st.session_state.chat_history), start=1):
                st.markdown(f"**Q{idx}:** {item['question']}")
                st.markdown(f"**A{idx} (RAG):** {item['answer']}")
                #lưu lịch sử câu trả lời của graph rag
                st.markdown(f"**A{idx} (Graph):** {item['answer_graph_rag']}")
                st.divider()
        else:
            st.info("No history yet.")

        if st.button("Clear History", type="secondary"):
            st.session_state.chat_history = []
            st.success("Chat history cleared.")

        if st.button("Clear Vector Store", type="secondary"):
            st.session_state.retriever = None
            st.session_state.vector_store = None
            st.session_state.active_file = None
            st.success("Vector store cleared.")

        if st.button("Clear Image OCR Context", type="secondary"):
            st.session_state.image_context = None
            st.session_state.image_file_key = None
            st.success("Image OCR context cleared.")

    return chunk_size, chunk_overlap, top_k, model_name, enable_graph_rag


def main() -> None:
    init_state()

    chunk_size, chunk_overlap, top_k, model_name, enable_graph_rag = render_sidebar()

    st.title("SmartDoc AI - Intelligent Document Q&A")
    st.write("Upload a PDF and ask questions about its content.")

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file is not None:
        file_key = f"{uploaded_file.name}:{uploaded_file.size}"
        should_process = st.session_state.active_file != file_key

        if should_process:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_pdf_path = tmp_file.name

            try:
                with st.spinner("Processing PDF: loading, splitting, and indexing..."):
                    retriever, vector_store, num_pages, num_chunks, chunks = build_retriever(
                        temp_pdf_path,
                        chunk_size,
                        chunk_overlap,
                        top_k,
                    )
                    
                    # Xử lý graph (có option bỏ qua)
                    if enable_graph_rag:
                        st.write("📊 Đang xử lý dữ liệu đồ thị...")
                        graph_processed = process_text_to_graph(chunks, graph)
                        if graph_processed > 0:
                            st.success(f"✅ Xử lý graph thành công: {graph_processed} documents")
                        else:
                            st.warning("⚠️ Không thể xử lý graph, nhưng Vector RAG vẫn hoạt động")
                    else:
                        st.info("ℹ️ Graph RAG disabled by user setting")

                st.session_state.retriever = retriever
                st.session_state.vector_store = vector_store
                st.session_state.active_file = file_key

                st.success(
                    f"✅ PDF indexed successfully. Pages: {num_pages}, chunks: {num_chunks}, top-k: {top_k}."
                )
            except Exception as exc:
                st.error(
                    "Failed to process document. Please ensure the PDF is valid and dependencies are installed."
                )
                st.exception(exc)
            finally:
                if os.path.exists(temp_pdf_path):
                    os.remove(temp_pdf_path)

    st.divider()
    st.subheader("Image OCR")
    st.write("Upload multiple images or a ZIP archive containing image files.")
    uploaded_images = st.file_uploader(
        "Upload image files or ZIP archive",
        type=["png", "jpg", "jpeg", "bmp", "gif", "tiff", "tif", "zip"],
        accept_multiple_files=True,
    )

    image_file_key = None
    if uploaded_images:
        image_file_key = compute_image_upload_key(uploaded_images)
        image_files = gather_images_from_uploaded_files(uploaded_images)
        if image_files and st.session_state.image_file_key != image_file_key:
            with st.spinner("Extracting text from images..."):
                extracted_texts = []
                for name, content in image_files:
                    try:
                        text = extract_text_from_image_bytes(content)
                    except Exception as exc:
                        text = f"[Lỗi khi đọc ảnh: {exc}]"
                    extracted_texts.append((name, text))

                if extracted_texts:
                    combined_text = "\n\n".join(
                        f"--- {name} ---\n{text or '[Không tìm thấy văn bản]'}"
                        for name, text in extracted_texts
                    )
                    st.session_state.image_context = combined_text
                    st.session_state.image_file_key = image_file_key
                    with st.expander("Preview OCR text"):
                        st.text_area("Extracted text", value=combined_text, height=300)

                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
                        for name, text in extracted_texts:
                            zipf.writestr(f"{Path(name).stem}.txt", text or "")
                    zip_data = zip_buffer.getvalue()

                    st.download_button(
                        "Download all extracted text as ZIP",
                        data=zip_data,
                        file_name="ocr_texts.zip",
                        mime="application/zip",
                    )

                    for name, text in extracted_texts:
                        st.download_button(
                            f"Download {Path(name).stem}.txt",
                            data=text.encode("utf-8"),
                            file_name=f"{Path(name).stem}.txt",
                            mime="text/plain",
                        )
        elif uploaded_images and not image_files:
            st.warning("No supported image files were found in the uploaded input.")
    elif st.session_state.image_context:
        st.info("Image content already loaded. Ask a question about uploaded images below.")

    question = st.text_input("Ask a question")
    question_about_image = st.text_input("Ask a question about uploaded images")

    if question_about_image:
        if not st.session_state.image_context:
            st.warning("Please upload and process image files first.")
        else:
            try:
                with st.spinner("Generating answer for image content..."):
                    image_answer = ask_image_question(
                        question_about_image,
                        st.session_state.image_context,
                        model_name,
                    )

                st.subheader("Answer about image content")
                st.write(image_answer)
            except Exception as exc:
                st.error("Could not generate answer from image content. Please ensure Ollama is running and the model is available.")
                st.exception(exc)

    if question:
        if st.session_state.retriever is None:
            st.warning("Please upload and process a PDF first.")
            return

        try:
            if enable_graph_rag:
                col1, col2 = st.columns(2)
            else:
                col1 = st.container()
                col2 = None
            
            with col1:
                with st.spinner("🔍 Vector RAG - generating answer..."):
                    answer_RAG, sources = ask_rag(question, st.session_state.retriever, model_name)

            if enable_graph_rag and col2:
                with col2:
                    with st.spinner("📊 Graph RAG - generating answer..."):
                        answer_GRAPH_RAG = ask_graph_rag(question, graph, chat_llm)
            else:
                answer_GRAPH_RAG = "[Graph RAG disabled]"

            if enable_graph_rag:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📄 Answer-RAG (Vector)")
                    st.write(answer_RAG)
                
                with col2:
                    st.subheader("🕸️ Answer-GRAPH-RAG")
                    st.write(answer_GRAPH_RAG)
            else:
                st.subheader("📄 Answer (Vector RAG)")
                st.write(answer_RAG)

            st.session_state.chat_history.append(
                {
                    "question": question,
                    "answer": answer_RAG,
                    "answer_graph_rag": answer_GRAPH_RAG if enable_graph_rag else None,
                }
            )

            if sources:
                st.subheader("📚 Sources (Vector RAG)")
                for i, src in enumerate(sources, start=1):
                    st.markdown(f"**Source {i}** - page: {src['page']}")
                    st.caption(src["snippet"])
        except Exception as exc:
            err_text = str(exc)
            if "requires more system memory" in err_text:
                st.error(
                    "Model is too large for current available RAM. "
                    "Please switch to a smaller model (qwen2.5:1.5b or qwen2.5:3b) in the sidebar."
                )
                st.info("Run in terminal: ollama pull qwen2.5:1.5b")
            elif "model" in err_text.lower() and "not found" in err_text.lower():
                st.error("Selected model is not available in Ollama.")
                st.info("Run in terminal: ollama pull <model_name>")
            else:
                st.error(
                    "Could not generate answer. Please check that Ollama is running and the model is available."
                )
                st.exception(exc)


if __name__ == "__main__":
    main()
