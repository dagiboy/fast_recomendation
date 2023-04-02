FROM python:3.10

RUN pip3 install kaggle tqdm
COPY kaggle.json /root/.kaggle/
COPY src/download_dataset.py .
RUN python download_dataset.py

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY src/process_data.py .
RUN python process_data.py

COPY src /app
WORKDIR /app

ENTRYPOINT ["python", "model.py"]
