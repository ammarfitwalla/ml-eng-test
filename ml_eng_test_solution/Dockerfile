# FROM anibali/pytorch:cuda-9.0
FROM python:3.7.16

RUN apt-get update
RUN apt-get upgrade -y
RUN apt install libgl1-mesa-glx -y
RUN apt install libgl1-mesa-dev -y
RUN apt-get install -y \
        build-essential 

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 3000
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "test_fastapi:app", "--host", "0.0.0.0", "--port", "3000", "--limit-concurrency", "50", "--limit-max-requests","1000"]