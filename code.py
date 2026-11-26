import os
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import ollama
from sentence_transformers import SentenceTransformer, util

# Step 1: Define what a Document looks like
class Document:
    def __init__(self, content: str, metadata: dict = None):
        self.content = content # Example value: content holds "Valorant is a free-to-play... Phoenix Blaze... Veto Evolution...


        self.metadata = metadata if metadata is not None else {}
        # Example value: metadata holds {"source": "game_info.txt", "char_count": 8509}
        # If someone passed None, it becomes an empty {}





    def __repr__(self):


        # Example output when you print: 
        # "Document(metadata={'source': 'game_info.txt', 'char_count': 8509}, preview='Valorant is a free-to-play first-person tactica...')"
        return f"Document(metadata={self.metadata}, preview='{self.content[:50]}...')"





# Step 2: The Loader Function
def load_text_file(file_path: str) -> Document:
    """Reads a text file and returns a Document object."""
    
    # Example: checks if "data/game_info.txt" exists. If someone typed "data/pubg.txt", it says False!
    if not os.path.exists(file_path):
        # Example error shown if file is missing: FileNotFoundError: File not found: data/game_info.txt
        raise FileNotFoundError(f"File not found: {file_path}")



    # Opens "data/game_info.txt" in read mode ("r") so we can safely read it without modifying
    with open(file_path, "r", encoding="utf-8") as f:
        # Example value: text variable now holds the whole 8,509 letters starting with "Valorant is a free-to-play..."
        text = f.read()
    # Create the metadata dictionary with real facts about this file
    metadata = {
        # os.path.basename("data/game_info.txt") strips "data/" and keeps only "game_info.txt"
        "source": os.path.basename(file_path),
        
        # len(text) counts all letters and spaces -> gives the exact number 8509
        "char_count": len(text)
    }
    # Bundles the text and metadata together and hands it back
    # Returns: Document(content="Valorant is a...", metadata={"source": "game_info.txt", "char_count": 8509})
    return Document(content=text, metadata=metadata)




# ==========================================
# STEP 2: THE TEXT SPLITTER (THE SCISSORS)
# ==========================================
def split_text(text: str, chunk_size: int = 300, chunk_overlap: int = 50) -> list[str]:
    """Cuts a long string of text into smaller chunks with overlap."""
    
    chunks = []  # A list that will hold all our sliced pieces
    start = 0    # Starting letter position (starts at 0)
    
    # Keep slicing as long as our starting point hasn't reached the end of the text
    while start < len(text):
        # Calculate the end position of this slice
        end = start + chunk_size
        
        # Grab the text slice from 'start' to 'end'
        chunk = text[start:end]
        
        # Add this slice to our list
        chunks.append(chunk)
        
        # Move forward! Notice: we move forward by (chunk_size - chunk_overlap)
        # Example: 500 - 50 = 450, so the next chunk starts 50 characters before this one ended!
        start += (chunk_size - chunk_overlap)
        
    return chunks










def build_prompt(question: str, chunks: list[str]) -> str:
     # Example input for 'question':
    #   question = "Which character can heal?"
    #
    # Example input for 'chunks' (a list with 2 text cards):
    #   chunks = [
    #       "Miks: M-pulse throws a device that can heal teammates.",
    #       "Sage: Healing Orb heals Sage or a teammate."
    #   ]





    # Glue the chunks together with an empty line between each
    context = "\n\n".join(chunks)
        # It glued the 2 cards together, putting an empty line (\n\n) between them!
    #
    # Example value of 'context' right now:
    # """Miks: M-pulse throws a device that can heal teammates.
    # 
    # Sage: Healing Orb heals Sage or a teammate."""





    prompt = f"""read all of the chunks and give the proper answer"

Context:
{context}

Question: {question}

Answer:"""

    return prompt






# STEP 5: THE LLM (THE SMART FRIEND)
# ==========================================
def generate_answer(prompt: str, model_name: str = "qwen3:4b") -> str:
    """Sends the prompt to Ollama and gets the answer back."""

    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        think=False,  # Tell qwen3 to answer directly, without "thinking out loud" first
    )

    # The reply is a box inside a box: open "message", then take "content"
    return response["message"]["content"]





if __name__ == "__main__":
    # 1. Load the document (Step 1)
    file_path = "data/game_info.txt"
    doc = load_text_file(file_path)
    #print(doc)

    # 2. Cut it into chunks (Step 2)
    chunks = split_text(doc.content, chunk_size=500, chunk_overlap=50)
    print(f"Total chunks created: {len(chunks)}")

    # 3. Load our embedding model (Step 3)
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 4. Turn ALL chunks into embeddings (numbers)!
    print("Calculating embeddings for all chunks...")
    chunk_embeddings = model.encode(chunks)

    print("\nReady! Ask me anything about Valorant. Type 'exit' to quit.")


        # 5. Keep asking questions until the user types exit
    while True:
        query = input("\nAsk: ").strip()

        if not query:
            continue
        if query.lower() == "exit":
            print("Bye!")
            break

        # Convert the question into numbers
        query_embedding = model.encode(query)

        # 6. RETRIEVAL: Compare question with all chunks
        similarity_scores = util.cos_sim(query_embedding, chunk_embeddings)[0]

        # 7. Find top 3 chunks
        top = torch.topk(similarity_scores, k=min(10, len(chunks)))

        top_chunks = []
        for score, index in zip(top.values, top.indices):
            print(f"  Found chunk {index.item()} (score: {score.item():.2f})")
            top_chunks.append(chunks[index.item()])

        # 8. Build the prompt with context
        prompt = build_prompt(query, top_chunks)
        print("Thinking...")

        # 9. Ask the LLM
        try:
            answer = generate_answer(prompt)
        except Exception as e:
            #print(f"Could not reach Ollama: {e}")
            print("Make sure the Ollama app is running and 'qwen3:4b' is downloaded.")
            continue

        # 10. Show the answer
        print(f"\nAnswer: {answer}")