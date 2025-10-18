from fastapi import APIRouter
from .geolocation import router as geolocation_router
from .discovery import router as discovery_router
from .signals import router as signals_router
from .auth import router as auth_router

api_router = APIRouter()

api_router.include_router(geolocation_router)
api_router.include_router(discovery_router) 
api_router.include_router(signals_router)
api_router.include_router(auth_router)