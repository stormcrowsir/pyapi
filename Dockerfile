FROM python:3.9-alpine

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
COPY ./google-cred.json /code/google-cred.json
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

CMD ["python", "app/main.py"]
