import jwt
from jwt import InvalidSignatureError, ExpiredSignatureError
from cryptography.hazmat.primitives import serialization


def load_public_key(path):
    with open(path, "rb") as f:
        return serialization.load_pem_public_key(f.read())


def verify_vc(vc_token, client_id):
    public_key_path = f"keys/{client_id}/public.pem"

    try:
        public_key = load_public_key(public_key_path)

        # Decode JWT with RSA public key
        payload = jwt.decode(
            vc_token,
            public_key,
            algorithms=["RS256"]
        )

        # Core validation
        if payload.get("role") != "authorized_fl_client":
            return False, "Invalid role"

        if payload.get("issuer") != "HealthAuthority":
            return False, "Invalid issuer"

        return True, payload

    except ExpiredSignatureError:
        return False, "Credential expired"

    except InvalidSignatureError:
        return False, "Invalid signature"

    except Exception as e:
        return False, f"Error: {str(e)}"


# Only run when testing manually
if __name__ == "__main__":
    with open("credentials/client1_vc.jwt") as f:
        token = f.read()

    ok, info = verify_vc(token, "client1")
    print("Valid?" , ok)
    print("Info:", info)
