from __future__ import annotations

import os
from typing import List

from langchain_core.documents import Document as LC_Document
from langchain_ollama import ChatOllama

from services.notebook_storage import NotebookStorage

from falkordb import FalkorDB
from pathlib import Path
import subprocess
import redis
import os
import time
# --- CONFIG ---
# FalkorDB mặc định chạy trên cổng 6379 của Redis
DEFAULT_FALKOR_HOST = os.getenv("FALKOR_HOST", "0.0.0.0")
DEFAULT_FALKOR_PORT = int(os.getenv("FALKOR_PORT", 6379)) 
GRAPH_NAME = "LLMs_RAG_Graph"



class GraphRAGService:
    def __init__(self, storage: NotebookStorage) -> None:
        self.storage = storage
        self.graph_name = "LLMs_RAG_Graph"
        
        # 1. Khởi tạo kết nối và gán giá trị mặc định cho self.graph
        try:
            self.db = FalkorDB(host=DEFAULT_FALKOR_HOST, port=DEFAULT_FALKOR_PORT)
            # Khởi tạo instance graph ngay lập tức
            self.graph = self.db.select_graph(self.graph_name)
            print(f"[GraphRAG] Connected to FalkorDB at {DEFAULT_FALKOR_HOST}:{DEFAULT_FALKOR_PORT}")
        except Exception as e:
            self.graph = None # Đảm bảo thuộc tính tồn tại để không bị lỗi "AttributeError"
            print(f"[GraphRAG ERROR] Connection failed: {e}")

        self.chat_llm = ChatOllama(model="qwen2.5:1.5b", temperature=0.1) 
    
    def export_falkor_snapshot(self, dest_path):
        # 1. Kết nối tới Redis/FalkorDB
        r = redis.Redis(host='127.0.0.1', port=6379)
        r.save()
        
        try:
            # Kiểm tra thông tin hệ thống
            info = r.info('persistence')
            bg_save_in_progress = info.get('rdb_bgsave_in_progress')

            if bg_save_in_progress:
                print("⏳ Một tiến trình lưu ngầm đang chạy. Đang đợi hoàn tất...")
                # Đợi cho đến khi bgsave_in_progress trả về 0
                while r.info('persistence').get('rdb_bgsave_in_progress'):
                    time.sleep(1)
            else:
                print("🚀 Đang yêu cầu tạo Snapshot mới...")
                r.bgsave() # Dùng bgsave để không treo server
                while r.info('persistence').get('rdb_bgsave_in_progress'):
                    time.sleep(1)

            # Tạo tên file và đường dẫn đầy đủ
            container_name = "falkordb"
            filename = f"graph_backup.rdb"
            full_dest_path = os.path.join(dest_path, filename)

            # Thực hiện lệnh copy
            exit_code = os.system(f"docker cp {container_name}:/data/data/dump.rdb {full_dest_path}")

            if exit_code == 0:
                # Lấy đường dẫn tuyệt đối để print ra terminal
                absolute_path = os.path.abspath(full_dest_path)
                print(f"✅ Xuất file thành công!")
                print(f"📍 Vị trí lưu: {absolute_path}")
            else:
                print(f"❌ Có lỗi xảy ra khi thực hiện lệnh docker cp.")

        except Exception as e:
            print(f"❌ Lỗi khi xuất file: {e}")

    def load_falkor_snapshot(self, file_path: Path, container_name="falkordb"):
        """
        Nạp lại file backup .rdb vào FalkorDB container.
        """
        backup_file_path = os.path.join(file_path, "graph_backup.rdb")
        if not os.path.exists(backup_file_path):
            print(f"❌ Không tìm thấy file backup tại: {backup_file_path}")
            return False

        try:
            print(f"🔄 Đang chuẩn bị nạp file: {backup_file_path}")

            # 1. Dừng container (Bắt buộc để thay thế file database)
            print("🛑 Đang dừng container...")
            subprocess.run(["docker", "stop", container_name], check=True)

            # 2. Chép file backup vào đúng vị trí và đổi tên thành dump.rdb
            # Dựa trên cấu trúc thư mục bạn gặp lúc nãy (/data/data/)
            remote_path = f"{container_name}:/data/data/dump.rdb"
            
            print(f"📂 Đang chép file vào container tại {remote_path}...")
            subprocess.run(["docker", "cp", backup_file_path, remote_path], check=True)

            # 3. Khởi động lại container
            print("🚀 Đang khởi động lại FalkorDB...")
            subprocess.run(["docker", "start", container_name], check=True)

            # 4. Đợi vài giây để database load xong dữ liệu vào RAM
            time.sleep(2)
            print("✅ Nạp dữ liệu hoàn tất! Bạn có thể kiểm tra lại Graph.")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Lỗi thực thi lệnh Docker: {e}")
            return False
        except Exception as e:
            print(f"❌ Lỗi không xác định: {e}")
            return False


    def build_graph_from_chunks(self, chunks: List[LC_Document],notebook_id) -> int:
        if self.graph is None:
            print("[GraphRAG ERROR] No database connection.")
            return 0
        
        dirs = self.storage.get_notebook_dirs(notebook_id)
        index_dir = dirs["graph"]
        index_file = os.path.join(index_dir, "graph_backup.rdb")

        # 1. KIỂM TRA FILE DATABASE
        # Đường dẫn này phải khớp với nơi bạn lưu snapshot (ví dụ: backups/graph_backup.rdb)
        # Hoặc kiểm tra trực tiếp trong container nếu cần
        backup_path = Path(index_file)

        if backup_path.exists():
            print(f"[GraphRAG] 📁 Tìm thấy file backup tại {backup_path}. Đang nạp dữ liệu cũ...")
            self.load_falkor_snapshot(backup_path)
        else:
            # Kiểm tra xem graph có dữ liệu không trước khi xóa
            try:
                # Thử lấy số lượng node hiện tại
                res = self.graph.query("MATCH (n) RETURN count(n)")
                count_nodes = res.result_set[0][0]
                
                if count_nodes > 0:
                    print(f"[GraphRAG] Đang xóa {count_nodes} nodes cũ...")
                    self.graph.query("MATCH (n) DETACH DELETE n")
                else:
                    print("[GraphRAG] Graph đã trống, không cần xóa.")
                    
            except Exception as e:
                # Nếu graph chưa từng được tạo, lệnh MATCH sẽ lỗi "empty key"
                # Chúng ta im lặng bỏ qua vì mục tiêu là có một graph trống
                print("[GraphRAG] Khởi tạo Graph mới lần đầu.")
        # 2. TIẾN HÀNH THÊM NODE
        count = 0
        # Chạy dòng này ở hàm __init__ hoặc trước khi loop qua các chunks
        try:
            self.graph.query("CREATE INDEX ON :Entity(id)")
            self.graph.query("CREATE INDEX ON :Document(id)")
            self.graph.query("CREATE INDEX ON :CONTEXT(id)")
        except:
            # Nếu index đã tồn tại, FalkorDB có thể báo lỗi, ta chỉ cần bỏ qua
            pass
        try:
            for idx, doc in enumerate(chunks):
                # Tạo ID duy nhất bằng cách kết hợp timestamp hoặc uuid để tránh trùng với dữ liệu cũ
                unique_suffix = int(time.time()) + idx
                ctx_id = f"CONTEXT_{unique_suffix}"
                doc_id = f"DOC_{unique_suffix}"
                
                content = doc.page_content.replace('"', "'") 
                
                query = """
                MERGE (d:Document {id: $doc_id})
                ON CREATE SET d.source = $source

                CREATE (c:CONTEXT {id: $ctx_id, content: $content, description: $doc_id, source: $source, index: $index}) 
                MERGE (d)-[:HAS_CONTEXT]->(c)
                """
                
                params = {
                    "ctx_id": ctx_id,
                    "content": content,
                    "doc_id": doc_id,
                    "source": doc.metadata.get("source_name", "Unknown"),
                    "index": idx,  # Lưu số thứ tự vòng lặp
                }
                self.graph.query(query, params)
                count += 1
                
            print(f"[GraphRAG] ✅ Successfully added {count} new nodes.")
            


            return count
        except Exception as e:
            print(f"[GraphRAG ERROR] Build failed at chunk {count}: {e}")
        return count
        
    def ask_graph_question(self, question: str) -> str:
        if self.graph is None: return "❌ Kết nối DB lỗi."


        try:
            # Tìm các context có nội dung liên quan đến câu hỏi (giản đơn)
            # Lưu ý: FalkorDB không có Full-text search mặc định như Neo4j, 
            # nên dùng CONTAINS hoặc lấy các node mới nhất.
            stopwords = {
                'làm', 'cách', 'có', 'là', 'được', 'gì', 'nào', 'nên', 'hãy', 'để', 
                'từ', 'và', 'hay', 'như', 'tại', 'với', 'trong', 'trên', 'dưới', 'về', 
                'sau', 'trước', 'này', 'kia', 'không', 'sao', 'ai', 'nước', 'người', 
                'điều', 'còn', 'việc', 'mà', 'nhất', 'thì'
            }
            raw_words = question.lower().split()
            keywords = []
            for w in raw_words:
                # Xóa sạch ký tự không phải chữ/số để an toàn cho Cypher
                import re
                clean_w = re.sub(r"[^a-v0-9àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệđìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]", "", w)
                
                if len(clean_w) > 2 and clean_w not in stopwords:
                    # Quan trọng: Escape dấu nháy nếu lỡ còn sót lại (double check)
                    keywords.append(clean_w.replace("'", "\\'"))

            print(f"[GraphRAG] Extracted keywords: {keywords}")

            if not keywords:
                return "❌ Không tìm thấy từ khóa ý nghĩa."

            # Step 2: Build WHERE clause an toàn
            where_clauses = []
            # Giới hạn 3 từ khóa quan trọng nhất để tránh query quá dài
            for kw in keywords[:3]:
                where_clauses.append(f"n.id CONTAINS '{kw}'")
                where_clauses.append(f"toLower(ctx.content) CONTAINS toLower('{kw}')")

            where_condition = " OR ".join(where_clauses)
            print(f"[GraphRAG] WHERE condition: {where_condition}...")
            
            # Kiểm tra xem người dùng có muốn tóm tắt không
            summary_triggers = ["tóm tắt", "tổng hợp", "ý chính", "nội dung chính"]
            is_summary_req = any(t in question.lower() for t in summary_triggers)

            if is_summary_req or not keywords:
                # Lấy 15 đoạn văn bản đầu tiên để tóm tắt
                query = """
                MATCH (ctx:CONTEXT)
                RETURN ctx.content as content, ctx.index as id
                ORDER BY ctx.index ASC
                LIMIT 15
                """
            else:
                # Logic tìm theo từ khóa (đã sửa để tìm đúng vào ctx.content)
                where_clauses = []
                for kw in keywords[:3]:
                    where_clauses.append(f"toLower(ctx.content) CONTAINS toLower('{kw}')")
                
                where_condition = " OR ".join(where_clauses)
                query = f"""
                MATCH (n)-[:HAS_CONTEXT]->(ctx:CONTEXT)
                WHERE {where_condition}
                RETURN ctx.content as content, ctx.index as id
                ORDER BY ctx.index ASC
                LIMIT 10
                """

            results = self.graph.query(query)

            print(f"{results.result_set}")
            context_parts = []
            results_list = []
            if results.result_set:
                for row in results.result_set:
                    results_list.append({
                        "id": row[1],      # ctx.id
                        "content": row[0]  # ctx.content
                    })
            results_list.sort(key=lambda x: x['id'])
            context_parts = [item['content'] for item in results_list]

            context_text = "\n---\n".join(context_parts)
            
            # Gửi cho LLM
            prompt = f"""Bạn là một chuyên gia phân tích dữ liệu và tóm tắt văn bản chuyên nghiệp.
            Nhiệm vụ của bạn là đọc kỹ tài liệu được cung cấp dưới đây và xử lý yêu cầu của người dùng một cách chính xác, logic và dễ hiểu.

            === NỘI DUNG TÀI LIỆU ===
            {context_text}

            === YÊU CẦU / CÂU HỎI ===
            {question}

            === HƯỚNG DẪN CHI TIẾT (BẮT BUỘC TUÂN THỦ) ===
            1. TUYỆT ĐỐI TRUNG THỰC: Chỉ sử dụng dữ kiện có trong "NỘI DUNG TÀI LIỆU". Tuyệt đối không tự suy diễn, bịa đặt (hallucinate) hoặc dùng kiến thức bên ngoài.
            2. XỬ LÝ KHI THIẾU THÔNG TIN: Nếu câu hỏi nằm ngoài phạm vi tài liệu, hãy trả lời đúng nguyên văn: "Tài liệu được cung cấp không đề cập đến thông tin này."
            3. HƯỚNG DẪN TÓM TẮT (Nếu yêu cầu là "tóm tắt", "ý chính", "tổng hợp"):
            - Cấu trúc rõ ràng: Bắt đầu bằng một câu tổng quan, sau đó chia thành các ý chính.
            - Nhấn mạnh thông tin lõi: Giữ lại các số liệu quan trọng, tên riêng, hoặc kết luận chính.
            4. HƯỚNG DẪN TRẢ LỜI CHI TIẾT (Nếu là câu hỏi cụ thể):
            - Trả lời trực tiếp vào trọng tâm câu hỏi.
            - Luôn kèm theo ví dụ, số liệu hoặc trích dẫn từ tài liệu để làm dẫn chứng.
            5. ĐỊNH DẠNG TRÌNH BÀY: Sử dụng Markdown một cách thông minh. Dùng in đậm (**chữ**) cho từ khóa quan trọng, và dùng gạch đầu dòng (-) hoặc đánh số (1,2,3) để nội dung dễ đọc, rành mạch.

            === TRẢ LỜI ===
            """
            print("Calling Ollama with model: qwen2.5:1.5b")
            print(f"{prompt}")
            response = self.chat_llm.invoke(prompt)
            return response.content
            
        except Exception as e:
            return f"❌ Lỗi truy vấn: {e}"

    def answer_question(self, notebook_id: str, question: str) -> str:
        if not self.storage.notebook_exists(notebook_id):
            return "Notebook không tồn tại."
        return self.ask_graph_question(question)