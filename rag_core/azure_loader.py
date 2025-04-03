from azure.storage.blob import BlobServiceClient
from pathlib import Path

def load_pdfs_from_blob(connection_str: str, container_name: str, download_dir="temp_pdfs") -> list[str]:
    blob_service_client = BlobServiceClient.from_connection_string(connection_str)
    container_client = blob_service_client.get_container_client(container_name)
    Path(download_dir).mkdir(exist_ok=True)
    downloaded_files = []

    for blob in container_client.list_blobs():
        if blob.name.endswith(".pdf"):
            file_path = Path(download_dir) / Path(blob.name).name
            with open(file_path, "wb") as file:
                stream = container_client.download_blob(blob.name)
                file.write(stream.readall())
            downloaded_files.append(str(file_path))

    return downloaded_files
