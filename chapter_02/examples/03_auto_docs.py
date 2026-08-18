from fastapi import FastAPI, status
from fastapi.responses import RedirectResponse

app = FastAPI()


@app.get("/", include_in_schema=False)
def docs_redirect_controller():
    return RedirectResponse(url="/docs", status_code=status.HTTP_303_SEE_OTHER)
