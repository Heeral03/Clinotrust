from fastapi import FastAPI

app = FastAPI(title="Clinotrust")


@app.get("/health")
def health():
    return {"status": "ok"}
