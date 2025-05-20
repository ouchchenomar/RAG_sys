import os
import requests
from tqdm import tqdm

def download_model(url, save_path):
    """
    Télécharger un modèle depuis une URL
    
    Args:
        url (str): URL du modèle
        save_path (str): Chemin où sauvegarder le modèle
    """
    # Créer le répertoire si nécessaire
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Télécharger le modèle
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(save_path, 'wb') as f, tqdm(
            desc=os.path.basename(save_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

if __name__ == "__main__":
    # URL du modèle TinyLlama GGUF (version quantifiée pour être plus légère)
    # Note: Remplacer par l'URL d'un modèle GGUF que vous souhaitez utiliser
    model_url = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    save_path = "./models/tinyllama.gguf"
    
    print(f"Téléchargement du modèle depuis {model_url}...")
    download_model(model_url, save_path)
    print(f"Modèle téléchargé et sauvegardé à {save_path}")