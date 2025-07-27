from functools import wraps
from flask import request, jsonify

from .jwt import verify_token

#* The middleware will execute for every route in the API
def check_auth(f):
  @wraps(f)
  def decorated(*args, **kwargs):
    try:
        headers_auth = request.headers.get('authorization')
        if headers_auth:
            print(headers_auth.split(' '))
            token = headers_auth.split(' ')[1]
            print(token)
            res = verify_token(token)
            if not res:
                return jsonify({'error': "Token invalid"}), 401
            else:
                return f(*args, **kwargs)
        else:
            return jsonify({'error': "There's no token to use the API"}), 401
    except Exception as e:
        return jsonify({'error': f'{e.__str__()}'}), 401
    
  return decorated