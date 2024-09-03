from typing import Annotated
import os
from dotenv import load_dotenv
from fastapi import Depends, APIRouter, FastAPI
from fastapi.security import HTTPBasic, HTTPBasicCredentials



app = FastAPI(
    openapi_prefix='/py'
    )

load_dotenv()
security = HTTPBasic()
prefix_router = APIRouter(prefix="/api/v1")

# Add the paths to the router instead

@prefix_router.get("/")
def init():
    return os.getenv("PASSWORD")


def isAuth(username, password):
    return ((os.getenv("FASTAPI_USERNAME")==username) and (os.getenv("FASTAPI_PASSWORD")==password))


@prefix_router.get("/auth")
def read_current_user(credentials: Annotated[HTTPBasicCredentials, Depends(security)]):
    if(not isAuth(credentials.username,credentials.password)):
        return '500 Not Authorized'
    return '200 success'


app.include_router(prefix_router)