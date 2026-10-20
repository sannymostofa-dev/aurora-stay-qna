from pathlib import Path
import fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR = Path("../data")
OUTPUT_DIR = Path("faiss_index")
EMBED_MODEL = "HarishMaths/Hotel-Policy-Embedding"
CHUNK_SIZE = 120
CHUNK_OVERLAP = 15

pdfs = sorted(DATA_DIR.glob("*.pdf"))
if not pdfs:
    raise FileNotFoundError("No PDF files found in ../data")

documents = []
page_count = 0
for pdf_path in pdfs:
    pdf = fitz.open(pdf_path)
    page_count += len(pdf)
    for page_number, page in enumerate(pdf, start=1):
        text = page.get_text("text").strip()
        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path.name,
                        "page": page_number,
                    },
                )
            )

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=lambda s: len(s.split()),
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_documents(documents)

print(f"Loaded {len(pdfs)} PDFs / {page_count} pages")
print(f"Created {len(chunks)} chunks")
print(f"Embedding model: {EMBED_MODEL}")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = FAISS.from_documents(chunks, embeddings)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
vectorstore.save_local(str(OUTPUT_DIR))

print(f"Saved FAISS index to: {OUTPUT_DIR.resolve()}")
print(f"Vectors stored: {vectorstore.index.ntotal}")
