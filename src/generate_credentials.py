import jwt
import json
from cryptography.hazmat.primitives import serialization
import time
import os


def load_private_key(path):
    with open(path, "rb") as f:
        private_key = serialization.load_pem_private_key(
            f.read(),
            password=None,
        )
    return private_key


def create_vc(client_id, private_key_path, output_path):
    private_key = load_private_key(private_key_path)

    # JWT payload (VC)
    payload = {
        "client_id": client_id,
        "role": "authorized_fl_client",
        "issuer": "HealthAuthority",
        "iat": int(time.time()),
        "exp": int(time.time()) + 9999999  # long expiry
    }

    # Sign using RS256
    token = jwt.encode(
        payload,
        private_key,
        algorithm="RS256"
    )

    # Save VC token
    with open(output_path, "w") as f:
        f.write(token)

    print(f"VC generated for {client_id} -> {output_path}")


# Create folders
os.makedirs("credentials", exist_ok=True)

# Generate VCs for all three clients
create_vc("client1", "keys/client1/private.pem", "credentials/client1_vc.jwt")
create_vc("client2", "keys/client2/private.pem", "credentials/client2_vc.jwt")
create_vc("client3", "keys/client3/private.pem", "credentials/client3_vc.jwt")
