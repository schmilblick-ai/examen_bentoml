# src/service.py
import json
import numpy as np
import pandas as pd
import bentoml
from bentoml.models import BentoModel
from pydantic import BaseModel, Field, ConfigDict
from starlette.responses import Response, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
import jwt
from datetime import datetime, timedelta


class Credentials(BaseModel):
    username: str
    password: str

class InputModel(BaseModel): 
    model_config = ConfigDict(populate_by_name=True)
    # Misère on doit mettre des alias
    GRE_Score:         float = Field(..., ge=260, le=340, alias="GRE Score")
    TOEFL_Score:       float = Field(..., ge=0,   le=120, alias="TOEFL Score")
    University_Rating: int   = Field(..., ge=1,   le=5,   alias="University Rating")
    SOP:               float = Field(..., ge=1,   le=5)
    LOR:               float = Field(..., ge=1,   le=5)
    CGPA:              float = Field(..., ge=0,   le=10)
    Research:          int   = Field(..., ge=0,   le=1)    
    
    
# Ne pas hardcoder les secrets en production
JWT_SECRET_KEY = "your_jwt_secret_key_here"
JWT_ALGORITHM = "HS256"

USERS = {
    "user123": "password123",
    "user456": "password456",
}

#ajout d'une classe pour distribuer le Bearer pour le test d'authentification via le swagger
class OpenAPISecurityMiddleware(BaseHTTPMiddleware):
    """Injecte le Bearer scheme dans le schéma OpenAPI généré par BentoML."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Intercepter uniquement la route du schéma OpenAPI
        if request.url.path not in ("/docs.json", "/openapi.json"):
            return response

        # Lire le body de la réponse
        body = b""
        async for chunk in response.body_iterator:
            body += chunk

        schema = json.loads(body)

        # Injecter le securityScheme Bearer
        schema.setdefault("components", {})
        schema["components"]["securitySchemes"] = {
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT"
            }
        }
        # Appliquer globalement à tous les endpoints
        schema["security"] = [{"BearerAuth": []}]

        patched = json.dumps(schema).encode()

        return Response(
            content     = patched,
            status_code = response.status_code,
            headers     = dict(response.headers) | {"content-length": str(len(patched))},
            media_type  = "application/json"
        )



def create_jwt_token(user_id: str) -> str:
    expiration = datetime.utcnow() + timedelta(hours=1)
    payload = {"sub": user_id, "exp": expiration}
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

class JWTAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.url.path.endswith("/predict"):
            token = request.headers.get("Authorization")
            if not token:
                return JSONResponse(status_code=401, content={"detail": "Missing authentication token"})
            try:
                token = token.split()[1]  # Bearer <token>
                payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            except jwt.ExpiredSignatureError:
                return JSONResponse(status_code=401, content={"detail": "Token has expired"})
            except jwt.InvalidTokenError:
                return JSONResponse(status_code=401, content={"detail": "Invalid token"})
            request.state.user = payload.get("sub")
        return await call_next(request)





@bentoml.service()
class ModelService:
    model_ref = BentoModel("admission_ridge:latest")

    def __init__(self):
        self.model = bentoml.sklearn.load_model(self.model_ref)
        # To be tested, it can be a non existing practice
        try:
            self.features = self.model_ref.info.metadata["features"]
        except BaseException as be:
            raise (f"Please save the feature list in the metadata model {be}") 

    @bentoml.api()
    def predict_input(self, input_data: InputModel) -> list[float]:
      #--- checking alignment of pydantic and model def
        #this only valid if input_data is considered a dict in def predict statement, not a pydantic
        #missing = set(self.features) - set(input_data.keys())
        #we retrieve the model_fields from the sub declared InputModel

        pydantic_input_fields = {
            info.alias or name
            for name, info in InputModel.model_fields.items()
        }
        model_features=set(self.features)
        print(model_features,pydantic_input_fields,sep="\n\n")
        missing_in_pydantic = model_features - pydantic_input_fields  # features du modèle absentes du schema
        extra_in_pydantic   = pydantic_input_fields - model_features    # champs Pydantic inconnus du modèle

        if missing_in_pydantic:
            raise RuntimeError(
                f"AdmissionInput manque ces features du modèle : {missing_in_pydantic}"
            )
        if extra_in_pydantic:
            raise RuntimeError(
                f"AdmissionInput a des champs inconnus du modèle : {extra_in_pydantic}"
            )
      # -- Si on arrive ici, schema Pydantic et modèle sont en phase --------
        print("✅ Schema Pydantic aligné avec les features du modèle")
       
      # Reconstruction dans l'ordre des features du training — peu importe l'ordre du dict entrant
        #Dict approach - sample = pd.DataFrame([[input_data[f] for f in self.features]],columns=self.features)  
        #pydantic approach - alignement du sample à prédir
        sample = pd.DataFrame([[getattr(input_data, f.replace(" ", "_")) for f in self.features]], columns=self.features )

        pred = self.model.predict(sample)
        return pred.tolist()


@bentoml.service()
class ModelProbaAdmission:
    model_service = bentoml.depends(ModelService)

    @bentoml.api(route="/login")
    def login(self, credentials: Credentials):
        if USERS.get(credentials.username) == credentials.password:
            token = create_jwt_token(credentials.username)
            return {"token": token}
        return JSONResponse(status_code=401, content={"detail": "Invalid credentials"})

    @bentoml.api(route="/predict")
    def predict(self, input_data: InputModel) -> dict:
        pred = self.model_service.predict_input(input_data)
        return {"prediction": pred}


# Appliquer le middleware sur tout le service
ModelProbaAdmission.add_asgi_middleware(JWTAuthMiddleware)
# Attacher le middleware au service
ModelProbaAdmission.add_asgi_middleware(OpenAPISecurityMiddleware)