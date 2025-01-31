# Conversational Recommender System Using a Combination of Fine-Tuned GPT-4o and Retrieval-Augmented Generation for Laptop Recommendations

## 📌 Project Overview
The **Conversational Recommender System Using a Combination of Fine-Tuned GPT-4o and Retrieval-Augmented Generation for Laptop Recommendations** is a **Conversational Recommender System (CRS)** that integrates **Fine-Tuned GPT-4o**, **Retrieval-Augmented Generation (RAG)** to provide accurate and scalable recommendations to user based on user's specifications.

## 🔥 Key Features
- **Conversational Interface:** Like other CRSes, users can express their preferences of a laptop in natural language.
- **Fine-Tuned GPT-4o:** A GPT-4o Model was trained to understand users specifications for certain laptops so it can take the intended specifications from the user and also give back a natural language response.
- **Retrieval-Augmented Generation (RAG):** Using RAG's Retrieval technique, we can retrieve laptops from a large dataset before generating responses.
- **Hybrid Approach:** The Combined Model (Fine-Tuned GPT-4o and RAG's Retrieval Technique) combines the power of generative ai with similiarity-based retrieval for improved accuracy.
- **FAISS Indexing:** Fast similarity searches among thousands of laptop entries.
- **Evaluation Metrics:** CRS Model was assessed using **Hit Rate, Precision, and NDCG**, achieving **high user satisfaction**.

## 🏆 Performance Summary
The CRS was evaluated using three models:
1. **RAG-based Model**
2. **Fine-Tuned GPT-4o Model**
3. **Combined Model (GPT-4o + RAG)**

| Metric  | RAG Model | Fine-Tuned GPT-4o | Combined Model |
|---------|------------|-------------------|----------------|
| **Hit Rate**  | 0.90  | 1.00  | 1.00  |
| **Precision** | 0.82  | 0.90  | 0.91  |
| **NDCG**      | 0.85  | 0.99  | 0.98  |
| **User Preference** | 15  | 21  | 60 |

The **Combined Model (GPT-4o + RAG)** outperformed other models as a whole, achieving **higher accuracy and user satisfaction.**

## 📂 Dataset Information
The system utilizes two primary datasets:
1. **Brands Laptops Dataset** (Kaggle) - Contains **991 laptops** with detailed specifications.
2. **Amazon Laptop Metadata Dataset** - Contains **26,546 laptops** after cleaning and preprocessing from Amazon Reviews Electronic Metadata Dataset.

🔗 **Download Datasets:** [Google Drive Link](https://drive.google.com/drive/folders/1qdMcCCsoEq3gtTwJ1vFWJTnmpT9dq73c?usp=sharing)

## 🚀 Installation & Setup
### 1️. Clone the Repository
git clone https://github.com/Zeltan13/LaptopCRS.git
cd LaptopCRS

### 2. Check the versions of libraries
- Python: 3.10
- openai: 1.54.4
- faiss: 1.7.4
- pandas: 2.0.3
- numpy: 1.26.1
- tqdm: 4.66.4
- json: Built-in module (no version)
- jsonlines: 4.0.0
- os: Built-in module (no version)
- sklearn: 1.5.1
'''bash 
pip install openai==1.54.4 faiss-cpu==1.7.4 pandas==2.0.3 numpy==1.26.1 tqdm==4.66.4 jsonlines==4.0.0 scikit-learn==1.5.1
## 3. Get your own OpenAI API
- Sign up/ Log in to OpenAI website
- Get Your API Key
- Replace "Your OpenAI API Key" on the code to your own OpenAI API Key 

## 4. Run the Recommender System
To run the **RAG-Based CRS**:
python RAG_CRS.py

To run the **Fine-Tuned GPT-4o CRS**:
python Fine_Tuned_GPT4o_CRS.py

To run the **Combined Model (RAG + GPT-4o) CRS**:
python Combined_Model_CRS.py

## 🔧 Configuration
- **FAISS Indexing:** The FAISS index (`faiss_index_new.bin`), is automatically generated and reused for efficiency.
- **Fine-Tuning GPT-4o:** The `Fine_Tuning_GPT4.ipynb` notebook guides you through fine-tuning process of the GPT-4o Model using OpenAI's website.
- **API Key Configuration:** Ensure you have set your OpenAI API key inside `Fine_Tuned_GPT4o_CRS.py` and `Combined_Model_CRS.py`: 
    client = OpenAI(api_key='Your OpenAI API Key')
- **Dataset Handling::** The metadata for the laptops is stored in metadata_cleaned.csv. Ensure it is located in the same directory before running the scripts.

## 🛠 Tech Stack
- Main Language: Python
- Retrieval: FAISS
- LLM Model Used: GPT-4o
- Text Processing: Scikit-Learn
- Data Handling: Pandas & NumPy

## 📈 Evaluation & User Feedback
- Performance Metrics Used: Hit Rate, Precision, and NDCG
- User Feedback: Conducted with 32 participants, with each participant trying out the three models on three seperate scenarios, picking their prefereed models 96 times, with Combined Model being prefereed 60 times

## 📜 License
- MIT Liscense