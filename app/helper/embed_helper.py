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
from langchain_community.chat_models import BedrockChat
from langchain.chains import LLMChain,ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_aws import BedrockLLM
import json
import os

dataset_path = 'hub://josuasirustara/education3'
embedding_id = "sentence-transformers/all-MiniLM-L6-v2"
repo_id = "HuggingFaceTB/SmolLM-135M"
model_id = "mistral.mistral-7b-instruct-v0:2"

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

        # llm = HuggingFaceEndpoint(
        #     repo_id=repo_id, max_length=128, temperature=0.5
        # )
        llm = BedrockChat(
            model_id=model_id,
            model_kwargs={'temperature': 0.5}
        )
        QA = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, return_source_documents=True)
    return QA

def generate_learning_path(topic):
    file_path = f"LLM-output-{topic}.json"
    if os.path.exists(file_path):
        # Open and read the JSON file
        with open(file_path, "r") as json_file:
            data = json.load(json_file)

            return data

    chat_history = []
    qa = search_db()
    result = qa({'question': 
                 """List all of the topic discussed with the resource related to """+topic,
                "chat_history": chat_history})

    source_documents = result['source_documents']
    ans = str(result['answer'])

    print(ans)
    print('=============================')

    template = """Create a learning plan based on these topic order it based on the difficulty:  {question}

    Please answer in this format [Number]. [Topic learning]: [What is it] - [Resource], as a list only maximum of 20 items, do not repeat yourself.
    """
    prompt = PromptTemplate(template=template, input_variables=["question"])
    # llm = HuggingFaceEndpoint(
    #         repo_id=repo_id, max_length=128, temperature=0.5
    #     )
    
    llm = BedrockLLM(
        model_id=model_id, region_name='ap-southeast-2'
    )
    qaa = LLMChain(prompt=prompt, llm=llm)
    learning_plans = qaa.run(ans)

    # result = str(result)
    # result = result.replace('Obat untuk ', '')
    # result = result.replace('Penyakit ', '')
    # result = result.replace('\n', '')
    # result = result.replace('\n2', '')
    # result = result.replace('\n6', '')



    print(result)
    print('=============================')


    sd = []
    for i in range(len(source_documents)):
        sd.append(dict(source_documents[i]))

    json_dictionary = {'source_documents': sd, 'path': str(learning_plans),'topics':str(ans)}
    json_object = json.dumps(json_dictionary, indent=4)

    with open(file_path, "w") as outfile:
        outfile.write(json_object)

    return {'source_documents': sd, 'path': str(learning_plans),'topics':str(ans)}