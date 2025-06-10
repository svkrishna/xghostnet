from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import api_router

app = FastAPI(title="Signal Classification API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(api_router) 