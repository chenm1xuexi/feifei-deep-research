import os

from chainlit.utils import mount_chainlit
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from common.log import logger

load_dotenv()

app = FastAPI(
    title="Feifei Deep Research API",
    description="API for Feifei",
    version="0.0.1",
)

allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "http://localhost:8888")
allowed_origins = [origin.strip() for origin in allowed_origins_str.split(",")]

logger.info(f"Allowed origins: {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Restrict to specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Use the configured list of methods
    allow_headers=["*"],  # Now allow all headers, but can be restricted further
)

# 添加chainlit路由
mount_chainlit(app=app, target="./chainlit_app/chainlit_api.py", path="")


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)


