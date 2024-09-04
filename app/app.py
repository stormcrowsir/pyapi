from typing import Annotated
import os
from dotenv import load_dotenv
from fastapi import Depends, APIRouter, FastAPI, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain.vectorstores import DeepLake
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from pydantic import BaseModel
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
import json
from fastapi.middleware.cors import CORSMiddleware

from helper.youtube_helper import fetch_youtube
from helper.udemy_helper import fetch_udemy
from helper.oreilly_helper import fetch_oreilly
# from helper.embed_helper import store_and_embed,search_db
from helper.extract_helper import extract_info

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"


app = FastAPI(
    openapi_prefix='/py'
    )

load_dotenv()
security = HTTPBasic()
prefix_router = APIRouter(prefix="/api/v1")

# Add the paths to the router instead

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class SearchRequest(BaseModel):
    query: str

@prefix_router.post("/search_youtube")
def search_youtube(request: SearchRequest):
    return fetch_youtube(request.query)


# @prefix_router.post("/embed_youtube")
# def embed_youtube(request: SearchRequest):
#     res = fetch_youtube(request.query)
#     texts = [x["title"] + ' ' + x['description'] + ' Published at ' +x['published_at'] for x in res["videos"]]
#     return store_and_embed(texts)

# @prefix_router.post("/get_db_embedding")
# def search_db_embedding(prompt: SearchRequest):
#     chat_history = []

#     qa = search_db()
#     result = qa({'question': prompt.query, "chat_history": chat_history})

#     return result['source_documents'] + "\nans : "+result['answer']

# def search_youtube(request: SearchRequest):
#     return fe(request.query)

@prefix_router.post("/search_udemy")
def search_udemy(request: SearchRequest):
    return fetch_udemy(request.query)


@prefix_router.post("/extract")
def extract_requirement(request: SearchRequest):
    return extract_info(request.query)

@prefix_router.post("/search_oreilly")
def search_oreilly(request: SearchRequest):
    return fetch_oreilly(request.query)

# @prefix_router.post("/ask")
# def question_answering(prompt: str) -> str:
    
#     chat_history = []
#     prompt = "Tuliskan semua penyakit dan masalah kesehatan dari semua dokumen yang diberikan"
#     qa = search_db()
#     result = qa({'question': prompt, "chat_history": chat_history})

#     source_documents = result['source_documents']
#     sickness = result['answer']

#     print(str(result['answer']))
#     print('=============================')

#     template = """Tuliskan semua obat untuk mengatasi penyakit yang muncul di paragraf:  {question}"""
#     prompt = PromptTemplate(template=template, input_variables=["question"])
#     llm = HuggingFaceEndpoint(
#             repo_id=repo_id, max_length=128, temperature=0.5
#         )
#     qaa = LLMChain(prompt=prompt, llm=llm)
#     result = qaa.run(str(result['answer']))

#     result = str(result)
#     result = result.replace('Obat untuk ', '')
#     result = result.replace('Penyakit ', '')
#     result = result.replace('\n', '')
#     result = result.replace('\n2', '')
#     result = result.replace('\n6', '')

#     drugs = result

#     print(result)
#     print('=============================')

#     template = """given Anatomical Therapeutic Chemical (ATC) Classification System categories:

#     1. M01AB - Anti-inflammatory and antirheumatic products, non-steroids, Acetic acid derivatives and related substances
#     2. M01AE - Anti-inflammatory and antirheumatic products, non-steroids, Propionic acid derivatives
#     3. N02BA - Other analgesics and antipyretics, Salicylic acid and derivatives
#     4. N02BE/B - Other analgesics and antipyretics, Pyrazolones and Anilides
#     5. N05B - Psycholeptics drugs, Anxiolytic drugs
#     6. N05C - Psycholeptics drugs, Hypnotics and sedatives drugs
#     7. R03 - Drugs for obstructive airway diseases
#     8. R06 - Antihistamines for systemic use


#     Help me categorize the drugs mentioned in this text: {question}
    
#     Please answer in this format [Number]. [Disease Name]: [ATC] - [Drug Name],"""

#     prompt = PromptTemplate(template=template, input_variables=["question"])
#     llm = HuggingFaceEndpoint(
#             repo_id=repo_id, max_length=128, temperature=0.5
#         )
#     qaa = LLMChain(prompt=prompt, llm=llm)
#     result = qaa.run(str(result))

#     sd = []
#     for i in range(len(source_documents)):
#         sd.append(dict(source_documents[i]))

#     json_dictionary = {'source_documents': sd, 'sickness': str(sickness), 'drugs': str(drugs), 'answer': str(result)}
#     json_object = json.dumps(json_dictionary, indent=4)

#     with open("dataset/LLM-output.json", "w") as outfile:
#         outfile.write(json_object)

#     return {'message': "successfully retrieve LLM answer in LLM-output.json"}

@prefix_router.get("/")
def init():
    return os.getenv("PASSWORD")


def isAuth(username, password):
    return ((os.getenv("FASTAPI_USERNAME")==username) and (os.getenv("FASTAPI_PASSWORD")==password))


# @prefix_router.get("/auth")
# def read_current_user(credentials: Annotated[HTTPBasicCredentials, Depends(security)]):
#     if(not isAuth(credentials.username,credentials.password)):
#         return '500 Not Authorized'
#     return '200 success'


app.include_router(prefix_router)