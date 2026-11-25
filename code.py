import os
os.environ["HF_HUB_OFFLINE"] = "1" 

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
def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
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










# Example: tells Python to import the neural network tool that translates text into numbers
from sentence_transformers import SentenceTransformer

# 1. Load a small, free embedding model (downloads once and stays on your computer)
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Let's take two sentences with similar meanings, and one completely different:
sentences = [
    "Phoenix throws a fireball that hurts enemies.",  # Fire / attack
    "Blaze makes a wall of flame and damages foes.",   # Same meaning as #1!
    "Sage heals her injured teammates with an orb."   # Completely different!
]

# 3. Convert these sentences into lists of numbers (embeddings)
embeddings = model.encode(sentences)

print(f"\nNumber of sentences: {len(embeddings)}")
print(f"Size of each embedding list: {len(embeddings[0])} numbers!")
print(f"First 5 numbers of Sentence 1: {embeddings[0][:5]}")




# Step 3: Test our loader
from sentence_transformers import SentenceTransformer, util

if __name__ == "__main__":
    # 1. Load the document (Step 1)
    file_path = "data/game_info.txt"
    doc = load_text_file(file_path)

    # 2. Cut it into chunks (Step 2)
    chunks = split_text(doc.content, chunk_size=500, chunk_overlap=50)
    print(f"Total chunks created: {len(chunks)}")

    # 3. Load our embedding model (Step 3)
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 4. Turn ALL chunks into embeddings (numbers)!
    print("Calculating embeddings for all chunks...")
    chunk_embeddings = model.encode(chunks)

    # 5. USER ASKS A QUESTION!
    query = "Which character can heal or revive a teammate?"
    print(f"\nUser Question: '{query}'")

    # Turn the user's question into 384 numbers too
    query_embedding = model.encode(query)






    # 6. RETRIEVAL: Compare the question with ALL chunks to find the closest match!
    # cos_sim calculates similarity scores between 0.0 and 1.0
    similarity_scores = util.cos_sim(query_embedding, chunk_embeddings)[0]
   

    # Find the chunk with the highest score!
    best_chunk_index = similarity_scores.argmax().item()
    
    best_score = similarity_scores[best_chunk_index].item()

    print(f"\n--- Best Matching Chunk Found! (Score: {best_score:.2f}) ---")
    print(chunks[best_chunk_index])