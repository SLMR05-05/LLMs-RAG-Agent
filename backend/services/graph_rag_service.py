from __future__ import annotations

import json
import os
from typing import List, Optional, Any

from langchain_core.documents import Document as LC_Document
from langchain_core.prompts import PromptTemplate
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_ollama import ChatOllama
from neo4j import GraphDatabase

from services.notebook_storage import NotebookStorage

# --- PROMPT TEMPLATES ---
CYPHER_GENERATION_TEMPLATE = """
Task: Convert the user's natural language question into a precise Cypher query.

Database Schema:
{schema}

Entity Attributes Reference:
- Use 'Title' for document or node headings.
- Use 'text' for core content retrieval.
- Supported properties: ['id', 'text', 'Title', 'source_name']

Constraint:
- Ensure property names are case-sensitive as defined above.
- If the node label is ambiguous, use a generic MATCH (n) pattern.

Question: {question}

Cypher Query:"""

CYPHER_PROMPT = PromptTemplate(
    input_variables=["schema", "question"],
    template=CYPHER_GENERATION_TEMPLATE,
)

QA_TEMPLATE = """
Task: Answer the user's question based strictly on the provided graph context.

Context (Graph Data):
{context}

User Question: {question}

Instructions:
1. Use ONLY the provided context to answer. 
2. If the context is empty or insufficient, state that the information is not available in the current relationship data.
3. Keep the answer concise and objective.

Answer:"""

QA_PROMPT_OBJ = PromptTemplate(input_variables=["context", "question"], template=QA_TEMPLATE)

# --- CONFIG ---
DEFAULT_NEO4J_URL = os.getenv("NEO4J_URL", "neo4j+s://a954d07c.databases.neo4j.io")
DEFAULT_NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
DEFAULT_NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "BNIJzfLaH-c5QOQqNAyS9igjQ15Ufe1uECTXYrVukL0")

def reset_database(uri, user, password):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        try:
            session.run("MATCH (n) DETACH DELETE n")
            print("Đã reset AuraDB thành công.")
        except Exception as e:
            print(f"Lỗi reset DB: {e}")
        finally:
            driver.close()

class GraphRAGService:
    def __init__(self, storage: NotebookStorage) -> None:
        self.storage = storage
        
        self.graph_db = Neo4jGraph(
            url=DEFAULT_NEO4J_URL,
            username=DEFAULT_NEO4J_USERNAME,
            password=DEFAULT_NEO4J_PASSWORD,
            database="neo4j"
        )

        # Note: Reset database during initialization is optional
        # Uncomment below to clear graph on startup
        # try:
        #     reset_database(DEFAULT_NEO4J_URL,
        #         DEFAULT_NEO4J_USERNAME,
        #         DEFAULT_NEO4J_PASSWORD)
        # except Exception as e:
        #     print(f"Warning: Could not reset database: {e}")
        
        self.chat_llm = ChatOllama(model="qwen2.5:1.5b", temperature=0.1) 
        
        self.graph_transformer = LLMGraphTransformer(
            llm=self.chat_llm,
            allowed_nodes=["Người", "Tổ_chức", "Công_nghệ", "Địa_điểm", "Dự_án"],
            allowed_relationships=["LÀ_THÀNH_VIÊN_CỦA", "SỬ_DỤNG", "PHÁT_TRIỂN", "LIÊN_QUAN_ĐẾN"]
        )
        print(f"[GraphRAG] Initialized with model: qwen2.5:1.5b")

    def build_graph_from_chunks(self, chunks: List[LC_Document]) -> int:
        """Build knowledge graph from chunks with content linking"""
        print(f"[GraphRAG] build_graph_from_chunks: {len(chunks)} chunks")
        if not chunks: 
            print(f"[GraphRAG ERROR] No chunks provided to build_graph_from_chunks")
            return 0
        
        count = 0
        try:
            # 0. Reset database before adding new data
            print(f"[GraphRAG] Resetting Neo4j database...")
            try:
                reset_database(DEFAULT_NEO4J_URL, DEFAULT_NEO4J_USERNAME, DEFAULT_NEO4J_PASSWORD)
                print("[GraphRAG] ✅ Database reset successfully")
            except Exception as e:
                print(f"[GraphRAG ERROR] Failed to reset database: {e}")
            
            # 1. Extract graph documents
            print(f"[GraphRAG] Extracting graph documents from {len(chunks)} chunks...")
            batch_size = 5 
            graph_docs_list = []
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                print(f"[GraphRAG] Processing batch {i//batch_size + 1}, chunk size: {len(batch[0].page_content) if batch else 0}")
                try:
                    graph_docs = self.graph_transformer.convert_to_graph_documents(batch)
                    print(f"[GraphRAG] Batch {i//batch_size + 1}: extracted {len(graph_docs) if isinstance(graph_docs, list) else 1} graph docs")
                    if graph_docs:
                        if not isinstance(graph_docs, list):
                            graph_docs = [graph_docs]
                        graph_docs_list.extend(graph_docs)
                except Exception as batch_err:
                    print(f"[GraphRAG ERROR] Failed to extract batch {i//batch_size + 1}: {str(batch_err)}")
            
            print(f"[GraphRAG] Total graph documents extracted: {len(graph_docs_list)}")
            
            # 1.5 FALLBACK: If LLM didn't extract entities, create them from chunks
            if not graph_docs_list:
                print(f"[GraphRAG WARNING] No entities extracted by LLM, using fallback: creating simple document nodes...")
                from langchain_core.documents import Document as GraphDocument
                fallback_docs = []
                for idx, chunk in enumerate(chunks):
                    # Create a simple document node with chunk content as description
                    source_name = chunk.metadata.get("source_name", f"Document_{idx}")
                    node_id = f"{source_name}_{idx}"
                    
                    # Create a document-like graph document
                    graph_doc = GraphDocument(
                        page_content=chunk.page_content,
                        metadata={
                            "id": node_id,
                            "description": chunk.page_content[:200],  # First 200 chars as description
                            "source": source_name,
                            "chunk_index": idx
                        }
                    )
                    fallback_docs.append(graph_doc)
                
                print(f"[GraphRAG] Created {len(fallback_docs)} fallback document nodes")
                # Don't add these directly - will handle in next step via CONTEXT nodes
                # These will act as context-only fallback
            
            # 2. Add graph documents to Neo4j
            if graph_docs_list:
                print(f"[GraphRAG] Adding {len(graph_docs_list)} graph documents to Neo4j...")
                try:
                    self.graph_db.add_graph_documents(graph_docs_list, include_source=True)
                    count = len(graph_docs_list)
                    print(f"[GraphRAG] ✅ Added {count} graph documents to Neo4j")
                except Exception as add_err:
                    print(f"[GraphRAG ERROR] Failed to add graph documents: {str(add_err)}")
            else:
                print(f"[GraphRAG WARNING] No graph documents - will use context-only retrieval")
            
            # 3. Add content contexts to entities
            print(f"[GraphRAG] Building content contexts from {len(chunks)} chunks...")
            full_content = "\n\n".join([doc.page_content for doc in chunks])
            print(f"[GraphRAG] Total content length: {len(full_content)} characters")
            
            # Split content into 5KB chunks with 500-char overlap
            content_chunks = []
            chunk_size = 5000
            overlap = 500
            
            for i in range(0, len(full_content), chunk_size - overlap):
                chunk = full_content[i:i + chunk_size]
                if chunk.strip():
                    content_chunks.append(chunk)
            
            print(f"[GraphRAG] Created {len(content_chunks)} content chunks (size: {chunk_size})")
            
            # Create CONTEXT nodes and link to all entities
            for chunk_idx, chunk_content in enumerate(content_chunks):
                ctx_id = f"CONTEXT_{chunk_idx}"
                
                try:
                    # Create CONTEXT node
                    print(f"[GraphRAG] Creating CONTEXT node {chunk_idx}...")
                    self.graph_db.query(
                        f"""
                        MERGE (ctx:CONTEXT {{id: $ctx_id}})
                        SET ctx.content = $content, ctx.chunk_index = $chunk_idx
                        """,
                        {"ctx_id": ctx_id, "content": chunk_content[:4900], "chunk_idx": chunk_idx}
                    )
                    print(f"[GraphRAG] ✅ CONTEXT node {chunk_idx} created")
                    
                    # Link ALL entities to this context
                    print(f"[GraphRAG] Linking entities to CONTEXT_{chunk_idx}...")
                    result = self.graph_db.query(
                        """
                        MATCH (ctx:CONTEXT {id: $ctx_id})
                        MATCH (e) WHERE NOT (e:CONTEXT)
                        MERGE (e)-[:HAS_CONTEXT]->(ctx)
                        RETURN count(e) as entity_count
                        """,
                        {"ctx_id": ctx_id}
                    )
                    entity_count = result[0].get('entity_count', 0) if result else 0
                    print(f"[GraphRAG] ✅ Linked {entity_count} entities to CONTEXT_{chunk_idx}")
                except Exception as ctx_err:
                    print(f"[GraphRAG ERROR] Failed to create/link context {chunk_idx}: {str(ctx_err)}")
            
            print(f"[GraphRAG] ✅ Graph processing complete: {count} documents, {len(content_chunks)} contexts")
            return count
            
        except Exception as e:
            print(f"[GraphRAG ERROR] Exception in build_graph_from_chunks: {type(e).__name__}: {str(e)}")
            import traceback
            print(f"[GraphRAG ERROR] Traceback:\n{traceback.format_exc()}")
            return 0

    def ask_graph_question(self, question: str, qa_prompt_input: Any = None) -> str:
        """Answer questions using graph entities with context and fallback mechanism"""
        try:
            print(f"[GraphRAG] Starting ask_graph_question with question: {question[:60]}...")
            
            # Step 1: Extract meaningful keywords with Vietnamese stopword filtering
            stopwords = {
                'làm', 'cách', 'có', 'là', 'được', 'gì', 'nào', 'nên', 'hãy', 'để', 
                'từ', 'và', 'hay', 'như', 'tại', 'với', 'trong', 'trên', 'dưới', 'về', 
                'sau', 'trước', 'này', 'kia', 'không', 'sao', 'ai', 'nước', 'người', 
                'điều', 'còn', 'việc', 'mà', 'nhất', 'thì'
            }
            keywords = [w.strip().lower() for w in question.split() 
                       if len(w.strip()) > 2 and w.strip().lower() not in stopwords]
            print(f"[GraphRAG] Extracted keywords: {keywords}")
            
            if not keywords:
                print(f"[GraphRAG] No meaningful keywords found")
                return "❌ Câu hỏi quá chung chung. Hãy dùng từ khóa cụ thể hơn (ví dụ: 'Django', 'AWS', 'project')."
            
            # Step 2: Build WHERE clause with CONTAINS operators (not LIKE or exact match)
            where_clauses = []
            for kw in keywords[:3]:
                where_clauses.append(f"n.id CONTAINS '{kw}'")
                where_clauses.append(f"n.description CONTAINS '{kw}'")
            
            where_condition = " OR ".join(where_clauses)
            print(f"[GraphRAG] WHERE condition: {where_condition[:100]}...")
            
            # Step 3: PRIMARY SEARCH - find entities matching keywords
            search_query = f"""
            MATCH (n) 
            WHERE {where_condition}
            OPTIONAL MATCH (n)-[:HAS_CONTEXT]->(ctx:CONTEXT)
            RETURN DISTINCT n.id as node_id, n.description as description, labels(n) as labels, 
                   collect({{id: ctx.id, content: ctx.content, idx: ctx.chunk_index}}) as contexts
            LIMIT 6
            """
            
            print(f"[GraphRAG] Executing primary search query...")
            results = self.graph_db.query(search_query)
            print(f"[GraphRAG] Primary search returned {len(results)} results")
            
            # Step 4: FALLBACK - if no keyword results, retrieve any entities with context
            if not results:
                print(f"[GraphRAG] No primary results, executing fallback query...")
                fallback_query = """
                MATCH (n)
                WHERE NOT (n:CONTEXT)
                OPTIONAL MATCH (n)-[:HAS_CONTEXT]->(ctx:CONTEXT)
                RETURN DISTINCT n.id as node_id, n.description as description, labels(n) as labels,
                       collect({id: ctx.id, content: ctx.content, idx: ctx.chunk_index}) as contexts
                LIMIT 6
                """
                results = self.graph_db.query(fallback_query)
                print(f"[GraphRAG] Fallback search returned {len(results)} results")
            
            # Step 5: Check if we have any results
            if not results:
                print(f"[GraphRAG] No entities found in graph, trying context-only fallback...")
                # Try to get all CONTEXT nodes as ultimate fallback
                try:
                    context_only_query = "MATCH (ctx:CONTEXT) RETURN ctx.id as ctx_id, ctx.content as content, ctx.chunk_index as idx LIMIT 10"
                    context_results = self.graph_db.query(context_only_query)
                    print(f"[GraphRAG] Found {len(context_results)} CONTEXT nodes")
                    
                    if context_results:
                        # Build response from contexts only
                        context_parts = []
                        for i, ctx in enumerate(context_results[:3], 1):
                            ctx_id = ctx.get('ctx_id', '')
                            content = ctx.get('content', '')
                            context_parts.append(f"[Context {i}] {ctx_id}")
                            if content:
                                context_parts.append(f"  {content[:500]}...")
                        
                        context_text = "\n".join(context_parts)
                        print(f"[GraphRAG] Using context-only mode with {len(context_results)} contexts")
                        
                        # Generate response from contexts
                        graph_prompt = f"""Bạn là chuyên gia phân tích tài liệu.
Dựa trên nội dung tài liệu dưới đây, hãy trả lời câu hỏi một cách chi tiết:

=== NỘI DUNG TÀI LIỆU ===
{context_text}

=== CÂU HỎI ===
{question}

=== HƯỚNG DẪN ===
1. Sử dụng CHỈ nội dung từ tài liệu, không bịa chuyện
2. Trả lời chi tiết với ví dụ từ tài liệu
3. Nếu tài liệu không đề cập, nói "Tài liệu không nêu"

=== TRẢ LỜI ==="""
                        
                        print(f"[GraphRAG] Invoking LLM with context-only prompt...")
                        response = self.chat_llm.invoke(graph_prompt)
                        
                        if hasattr(response, 'content'):
                            result = response.content
                        else:
                            result = str(response)
                        
                        print(f"[GraphRAG] ✅ Context-only response generated, length: {len(result)}")
                        return result
                except Exception as ctx_fallback_err:
                    print(f"[GraphRAG ERROR] Context-only fallback failed: {ctx_fallback_err}")
                
                print(f"[GraphRAG ERROR] No entities and no contexts found in graph")
                return f"❌ Không tìm được dữ liệu trong graph. Vui lòng upload PDF và xử lý lại."
            
            # Get all available contexts as secondary fallback
            try:
                all_contexts_query = "MATCH (ctx:CONTEXT) RETURN ctx.id as ctx_id, ctx.content as content, ctx.chunk_index as idx LIMIT 10"
                all_contexts = self.graph_db.query(all_contexts_query)
                print(f"[GraphRAG] Found {len(all_contexts)} total contexts in graph")
            except Exception as ctx_err:
                print(f"[GraphRAG] Failed to fetch all contexts: {ctx_err}")
                all_contexts = []
            
            # Step 6: Build comprehensive context from entities and their contexts
            context_parts = []
            for i, r in enumerate(results[:3], 1):  # Top 3 entities
                node_id = r.get('node_id', '')
                description = r.get('description', '')
                labels = r.get('labels', [])
                contexts = r.get('contexts', []) or []
                
                print(f"[GraphRAG] Processing entity {i}: {node_id}, labels={labels}, contexts_count={len(contexts)}")
                
                # Filter out empty contexts
                contexts = [c for c in contexts if c and c.get('content')]
                
                # FALLBACK: Assign available contexts if entity lacks them
                if not contexts and all_contexts:
                    contexts = all_contexts[:2]
                    print(f"[GraphRAG] Using fallback contexts for entity {i}")
                
                # Format entity header
                label_str = ', '.join(labels) if labels else 'Entity'
                context_parts.append(f"[Entity {i}: {label_str}] {node_id}")
                
                # Add description if available
                if description:
                    context_parts.append(f"  Description: {description[:500]}")
                
                # Add context chunks (top 2)
                if contexts:
                    context_parts.append(f"  Content chunks ({len(contexts)}):")
                    for j, ctx in enumerate(contexts[:2], 1):
                        content = ctx.get('content', '')[:600]
                        context_parts.append(f"    [{j}] {content}...")
            
            context_text = "\n".join(context_parts)
            print(f"[GraphRAG] Built context text, length: {len(context_text)}")
            
            # Step 7: Generate comprehensive LLM response with detailed prompt
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
            
            print(f"[GraphRAG] Calling LLM with prompt length: {len(graph_prompt)}")
            response = self.chat_llm.invoke(graph_prompt)
            print(f"[GraphRAG] LLM response received, type: {type(response)}")
            
            if hasattr(response, 'content'):
                result = response.content
                print(f"[GraphRAG] Extracted content from response, length: {len(result)}")
                return result
            
            result = str(response)
            print(f"[GraphRAG] Converted response to string, length: {len(result)}")
            return result
                
        except Exception as e:
            print(f"[GraphRAG ERROR] Exception in ask_graph_question: {type(e).__name__}: {str(e)}")
            import traceback
            print(f"[GraphRAG ERROR] Traceback:\n{traceback.format_exc()}")
            return f"❌ Lỗi xử lý graph: {str(e)[:100]}"

    def answer_question(self, notebook_id: str, question: str, prompt_input: Any = None) -> str:
        """Answer question using graph RAG with smart fallback"""
        if not self.storage.notebook_exists(notebook_id):
            return "Notebook không tồn tại."
        
        # Use improved ask_graph_question with fallback mechanism
        answer = self.ask_graph_question(question=question)
        return answer