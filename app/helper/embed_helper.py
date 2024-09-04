from deeplake.core.vectorstore import VectorStore
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from pydantic import BaseModel
from langchain_community.llms import HuggingFaceEndpoint
from langchain.vectorstores import DeepLake
from deeplake.core.vectorstore import VectorStore
from transformers import AutoTokenizer, AutoModel
from langchain_community.embeddings import HuggingFaceEmbeddings


dataset_path = 'hub://josuasirustara/education3'
embedding_id = "sentence-transformers/all-MiniLM-L6-v2"
repo_id = "HuggingFaceTB/SmolLM-135M"
QA = None
#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


tokenizer = AutoTokenizer.from_pretrained(embedding_id)
model = AutoModel.from_pretrained(embedding_id)


def embedding_function(texts):
    if isinstance(texts, str):
        texts = [texts]
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings


def store_and_embed(texts, chunk_size=1000):
    # texts = [x["title"] + ' ' + x['description'] + ' Published at ' +x['published_at'] for x in res["videos"]]

    vector_store = VectorStore(
        path=dataset_path,
    )

    for t in texts:
        chunked_text = [t[i:i+1000] for i in range(0, len(t), chunk_size)]

        vector_store.add(text=chunked_text,
                         embedding_function=embedding_function,
                         embedding_data=chunked_text,
                         metadata=[{"source": t}]*len(chunked_text))
        
    return { 
            'message':'success', 
            'status':200 
            }

def search_db():
    global QA

    if QA is None:
        embeddings = HuggingFaceEmbeddings(model_name=embedding_id)
        db = DeepLake(dataset_path=dataset_path, read_only=True, embedding=embeddings)
        retriever = db.as_retriever()
        retriever.search_kwargs['distance_metric'] = 'cos'
        retriever.search_kwargs['fetch_k'] = 50
        retriever.search_kwargs['k'] = 10

        llm = HuggingFaceEndpoint(
            repo_id=repo_id, max_length=128, temperature=0.5
        )
        QA = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, return_source_documents=True)

    return QA