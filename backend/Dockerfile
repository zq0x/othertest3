FROM python:3.12-slim

WORKDIR /usr/src/app

COPY . .

RUN pip install -r requirements.txt

EXPOSE 7861

CMD ["python", "backend.py"]
