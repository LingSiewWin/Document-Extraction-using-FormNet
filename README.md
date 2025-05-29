FormNet Document Extraction
Overview
FormNet is a document extraction model designed to handle complex form layouts, such as tables and columns, by leveraging structural information. It introduces a Rich Attention (RichAtt) mechanism that uses 2D spatial relationships between word tokens to compute accurate attention weights. Additionally, it employs a Graph Convolutional Network (GCN) to create Super-Tokens that embed contextual information from neighboring tokens. This implementation uses PyTorch, DGL, and Hugging Face Transformers to replicate FormNet's core components, targeting performance on benchmarks like CORD, FUNSD, and Payment datasets.
This project is a partial implementation inspired by the FormNet paper, focusing on the GCN-based Super-Tokens, Rich Attention mechanism, and integration with an Extended Transformer Construction (ETC) backbone. Note that some components are approximated due to the lack of publicly available source code.
Features

OCR Integration: Extracts words and 2D coordinates from form documents using Tesseract (extendable to other OCR engines).
GCN for Super-Tokens: Builds a graph of spatially related tokens and aggregates features using a GCN.
Rich Attention: Computes attention scores with a spatial penalty based on token coordinates.
ETC Transformer: Uses a transformer backbone for sequence modeling and entity extraction.
Viterbi Decoding: Placeholder for decoding entity sequences (to be implemented with libraries like torchcrf).

Prerequisites

Python: 3.8 or higher
Libraries:
PyTorch (pip install torch)
DGL (pip install dgl)
Transformers (pip install transformers)
Pytesseract (pip install pytesseract)
NumPy (pip install numpy)
Tesseract OCR (install system-wide, e.g., apt-get install tesseract-ocr on Ubuntu)


Hardware: GPU recommended for training.
Datasets: Access to CORD, FUNSD, and Payment datasets (available via academic repositories or Hugging Face Datasets).

Installation

Clone the repository:git clone <repository-url>
cd formnet-document-extraction


Install dependencies:pip install -r requirements.txt

Example requirements.txt:torch>=2.0.0
dgl>=1.0.0
transformers>=4.30.0
pytesseract>=0.3.10
numpy>=1.20.0


Install Tesseract OCR:
On Ubuntu: sudo apt-get install tesseract-ocr
On macOS: brew install tesseract
On Windows: Download and install from Tesseract GitHub.



Usage

Prepare a Document:

Place a form document (e.g., PDF or image) in the project directory.
Example: sample_form.pdf


Run the Model:

The main script (formnet.py) processes the document, extracts tokens, builds a graph, and runs the FormNet model.
Example command:python formnet.py --document sample_form.pdf


This will:
Extract tokens and coordinates using Tesseract.
Build a spatial graph and compute Super-Tokens.
Apply Rich Attention and ETC transformer.
Output entity predictions (requires labeled data for training).




Training:

Modify the script to include a training loop with your dataset.
Example datasets: CORD, FUNSD, Payment (pre-process into tokens, coordinates, and labels).
Adjust hyperparameters in the script (e.g., learning rate: 0.0002, GCN layers: 2).


Evaluation:

Evaluate on benchmarks using F1 score.
Compare against baselines like DocFormer (FormNet-A3 achieves 97.28% F1 on CORD).



Project Structure

formnet.py: Main script with FormNet model, GCN, and Rich Attention implementation.
data/: Directory for input documents and datasets.
models/: Directory for saving trained models.
requirements.txt: List of dependencies.

Limitations

Incomplete Implementation: Some components (e.g., Viterbi decoding, full pre-training) are placeholders due to missing details from the original FormNet paper.
OCR Dependency: Tesseract may not handle complex forms well; consider integrating Google Cloud Vision or Azure OCR for better accuracy.
Dataset Access: Requires access to CORD, FUNSD, and Payment datasets, which may need academic or institutional permissions.
Pre-training: The current code does not include the Masked Language Modeling (MLM) pre-training on 700k form documents, which is critical for FormNet’s performance.

Future Improvements

Implement full Viterbi decoding for entity sequence prediction.
Integrate a production-grade OCR engine (e.g., Google Cloud Vision).
Add pre-training with MLM on a large form dataset.
Optimize the GCN and Rich Attention for specific benchmarks.
Add support for batch processing and multi-GPU training.

References

FormNet Paper: [Link to paper, if available]
Datasets: CORD, FUNSD, Payment (check Hugging Face Datasets or academic repositories).
Libraries: PyTorch, DGL, Hugging Face Transformers, Pytesseract.

Contributing
Contributions are welcome! Please submit a pull request or open an issue for bugs, feature requests, or improvements.
License
MIT License
