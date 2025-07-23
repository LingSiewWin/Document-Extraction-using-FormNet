#🧾 FormNet-Based Document Extraction System
A partial implementation of FormNet – a powerful document information extraction model designed for structured documents like invoices, receipts, and forms. This project integrates OCR and graph-based learning to extract contextual entities by modeling both the layout and content of documents.

##📌 Overview
FormNet enhances traditional document parsing by introducing:

🧠 Rich Attention Mechanism: Computes attention using spatial relationships between word tokens.
##🔗 Graph Convolutional Network (GCN): Constructs Super-Tokens by aggregating neighboring contextual information.

⚙️ ETC Transformer Backbone: Utilizes sequence modeling for accurate entity extraction.

This implementation focuses on GCN Super-Tokens, Rich Attention, and basic ETC integration, with OCR support via Tesseract.

##🚀 Features
🔍 OCR Integration: Extracts words and coordinates using Tesseract OCR.

🧩 GCN-Based Super-Tokens: Graph modeling of spatially related tokens for better context.

📐 Rich Attention: Attention scoring with 2D layout awareness.

🧠 ETC Transformer: Captures relationships across the token graph.

🧪 Placeholder for Viterbi Decoding: Ready for sequence label decoding integration.

##🛠️ Setup
✅ Prerequisites
Python ≥ 3.8

Install the required libraries:

##bash
Copy
Edit
pip install -r requirements.txt
requirements.txt:

##shell
Copy
Edit
torch>=2.0.0  
dgl>=1.0.0  
transformers>=4.30.0  
pytesseract>=0.3.10  
numpy>=1.20.0  
🖥️ Install Tesseract OCR
macOS: brew install tesseract

Ubuntu: sudo apt-get install tesseract-ocr

Windows: Install from GitHub

###📄 Usage
1. Prepare Your Document
Place a sample form PDF or image (e.g., sample.pdf) in the project root.

###2. Run the Extraction
bash
Copy
Edit
python main.py --document sample.pdf
This will:

Use Tesseract to extract tokens and bounding boxes.

Build a spatial graph.

Apply the GCN + Rich Attention + ETC Transformer.

Output entity predictions (labels if available).

##🧪 Training (To Be Extended)
To train on your dataset (e.g., CORD, FUNSD):

Pre-process into token, coordinate, and label format.

Modify main.py to include a training loop.

Adjust hyperparameters (e.g., learning rate = 0.0002, GCN layers = 2).

##📊 Evaluation
Evaluate using standard metrics like F1-score, and benchmark against models like:

DocFormer

LayoutLM

FormNet-A3 (97.28% F1 on CORD)

##📁 Project Structure
bash
Copy
Edit
├── main.py                   # Main script
├── sample.pdf                # Input form
├── output.txt                # Prediction output
├── requirements.txt          # Dependencies
├── README.md                 # Project documentation
└── tempCodeRunnerFile.py     # Temporary script (optional)
⚠️ Limitations
❗ Partial Implementation – Viterbi decoding & pre-training are not included.

📄 OCR Quality – Tesseract may fail on noisy scans; better OCR engines recommended.

📚 Dataset Access – Requires datasets like CORD, FUNSD, Payment (available on Hugging Face or request-based academic access).

📉 No Pre-training – Lacks MLM-based large corpus pre-training, crucial for optimal performance.

##📈 Future Work
✅ Integrate Viterbi decoding for sequential labeling.

🔍 Replace Tesseract with Google Vision or Azure OCR.

🔁 Add Masked Language Modeling (MLM) pre-training on form datasets.

⚡ Optimize GCN/Attention for speed and accuracy.

🔄 Enable batch processing and multi-GPU training.

##📚 References
FormNet Paper: [Add Link Here]

Datasets: CORD, FUNSD, Payment

Libraries: PyTorch, DGL, Hugging Face Transformers, Pytesseract

##🤝 Contributions
Contributions are welcome!
Please open an issue or submit a pull request to help improve this project.

##📄 License
This project is licensed under the MIT License.