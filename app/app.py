from typing import Annotated
import os
from dotenv import load_dotenv
from fastapi import Depends, FastAPI
from fastapi.security import HTTPBasic, HTTPBasicCredentials

app = FastAPI(
    version=0.1,
    root_path="/pyapi/v1"
)
load_dotenv()
security = HTTPBasic()

@app.get("/")
def init():
    return os.getenv("PASSWORD")


def isAuth(username, password):
    return ((os.getenv("FASTAPI_USERNAME")==username) and (os.getenv("FASTAPI_PASSWORD")==password))


@app.get("/auth")
def read_current_user(credentials: Annotated[HTTPBasicCredentials, Depends(security)]):
    if(not isAuth(credentials.username,credentials.password)):
       return '500 Not Authorized'
    return '200 success'


