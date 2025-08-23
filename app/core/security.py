from fastapi import Header, HTTPException, status, Depends, Request
from typing import Optional
import os
import jwt
from functools import wraps

def verify_token(token: str):
    try:
        payload = jwt.decode(
            token,
            os.getenv('SECRET_KEY_TOKENS'),
            algorithms=['HS256']
        )
        return True if payload else False
    except jwt.ExpiredSignatureError:
        raise Exception('The token has expired or is invalid')
    except jwt.InvalidTokenError:
        raise Exception('The token is invalid')

async def check_auth(request: Request):
    headers_auth = request.headers.get('authorization')
    if headers_auth:
        try:
            parts = headers_auth.split(' ')
            if len(parts) != 2 or parts[0].lower() != 'bearer':
                raise HTTPException(status_code=401, detail="Invalid authorization header format")
            token = parts[1]
            res = verify_token(token)
            if not res:
                raise HTTPException(status_code=401, detail="Token invalid")
        except Exception as e:
            raise HTTPException(status_code=401, detail=str(e))
    else:
        raise HTTPException(status_code=401, detail="There's no token to use the API")
    return True
