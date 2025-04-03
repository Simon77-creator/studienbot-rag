# azure_loader.py
from azure.storage.blob import BlobServiceClient
import os

def load_pdfs_from_blob(connection_string, container_name):
    service = BlobServiceClient.from_connection_string(connection_string)
    container = service.get_container_client(container_name)

    local_dir = "/tmp/azure_pdfs"
    os.makedirs(local_dir, exist_ok=True)

    paths = []
    for blob in container.list_blobs():
        if blob.name.endswith(".pdf"):
            local_path = os.path.join(local_dir, os.path.basename(blob.name))
            with open(local_path, "wb") as f:
                data = container.download_blob(blob.name).readall()
                f.write(data)
            paths.append(local_path)

    return paths
