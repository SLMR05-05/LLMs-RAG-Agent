import hashlib
import os
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

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
	return retriever, vector_store, len(docs), len(chunks)

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


def render_sidebar() -> Tuple[int, int, int, str]:
	with st.sidebar:
		st.title("SmartDoc AI")
		st.caption("RAG system for PDF Q&A")

		st.subheader("Settings")
		chunk_size = st.slider("Chunk size", min_value=500, max_value=2000, value=1000, step=100)
		chunk_overlap = st.slider("Chunk overlap", min_value=50, max_value=300, value=100, step=25)
		top_k = st.slider("Top-k retrieval", min_value=1, max_value=10, value=3, step=1)
		model_name = st.text_input("Ollama model", value="qwen2.5:1.5b")
		st.caption("Low-RAM recommendation: qwen2.5:1.5b or qwen2.5:3b")

		st.subheader("Chat History")
		if st.session_state.chat_history:
			for idx, item in enumerate(reversed(st.session_state.chat_history), start=1):
				st.markdown(f"**Q{idx}:** {item['question']}")
				st.markdown(f"**A{idx}:** {item['answer']}")
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

	return chunk_size, chunk_overlap, top_k, model_name


def main() -> None:
	init_state()

	chunk_size, chunk_overlap, top_k, model_name = render_sidebar()

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
					retriever, vector_store, num_pages, num_chunks = build_retriever(
						temp_pdf_path,
						chunk_size,
						chunk_overlap,
						top_k,
					)

				st.session_state.retriever = retriever
				st.session_state.vector_store = vector_store
				st.session_state.active_file = file_key

				st.success(
					f"PDF indexed successfully. Pages: {num_pages}, chunks: {num_chunks}, top-k: {top_k}."
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
			with st.spinner("Generating answer..."):
				answer, sources = ask_rag(question, st.session_state.retriever, model_name)

			st.subheader("Answer")
			st.write(answer)

			st.session_state.chat_history.append(
				{
					"question": question,
					"answer": answer,
				}
			)

			if sources:
				st.subheader("Sources")
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
