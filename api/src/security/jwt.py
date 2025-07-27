import jwt
import os

#*The ANN API will only manage the token decoding and verify if the token is valid
#*The token encoding will be done by the web-app when the user logs in

def verify_token(token):
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