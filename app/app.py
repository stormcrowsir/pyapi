from typing import Annotated
import os
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi


app = FastAPI(
    version=0.1

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

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html(req: Request):
    root_path = req.scope.get("root_path", "").rstrip("/")
    openapi_url = root_path + app.openapi_url
    # print(openapi_url)

    return get_swagger_ui_html(
        openapi_url=openapi_url,
        title="API",
    )


