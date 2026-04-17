FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ❌ Removed spacy model download (not needed)

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]