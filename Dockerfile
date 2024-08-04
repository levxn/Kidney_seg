FROM python:3.8.18-slim

WORKDIR /app
COPY . .
RUN apt-get update
RUN apt install -y libglib2.0-0
RUN apt install -y libgl1-mesa-glx
COPY requirements.txt .
RUN pip install -r requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

CMD [ "python", "./docker.py" ]