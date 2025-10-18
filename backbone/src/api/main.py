from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import api_router
from .core.security import verify_token, security_manager
from fastapi.security import OAuth2PasswordBearer

app = FastAPI(
    title="GhostNet API",
    version="0.1.0",
    description="Passive RF mapping and mobility intelligence API",
    openapi_tags=[
        {"name": "auth", "description": "Authentication endpoints"},
        {"name": "geolocation", "description": "Geolocation and fingerprinting"},
        {"name": "signals", "description": "Signal spectrum and features"},
        {"name": "discovery", "description": "Device discovery"},
    ],
)

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


@app.get("/health")
async def health():
    return {"status": "ok"}

# Apply global security: all endpoints require bearer unless explicitly public
app.openapi_schema = None
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = app.openapi()
    # Inject global security requirement
    openapi_schema.setdefault("components", {}).setdefault("securitySchemes", {}).update({
        "OAuth2PasswordBearer": {
            "type": "oauth2",
            "flows": {
                "password": {
                    "tokenUrl": "/auth/token",
                    "scopes": {}
                }
            }
        }
    })
    openapi_schema["security"] = [{"OAuth2PasswordBearer": []}]
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

