import os
from typing import List, Dict, Any
import PyPDF2
import uuid

class DocumentLoader:
    """
    Classe pour charger des documents de différents formats (TXT, PDF)
    """
    
    def __init__(self, storage_dir: str = "./data/documents"):
        """
        Initialiser le chargeur de documents
        
        Args:
            storage_dir (str): Répertoire de stockage des documents
        """
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
    
    def load_text(self, file_path: str) -> str:
        """
        Charger un fichier texte
        
        Args:
            file_path (str): Chemin vers le fichier texte
            
        Returns:
            str: Contenu du fichier
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def load_pdf(self, file_path: str) -> str:
        """
        Charger un fichier PDF
        
        Args:
            file_path (str): Chemin vers le fichier PDF
            
        Returns:
            str: Contenu du fichier
        """
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
        return text
    
    def save_file(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Sauvegarder un fichier sur le disque
        
        Args:
            file_content (bytes): Contenu du fichier
            filename (str): Nom du fichier
            
        Returns:
            Dict[str, Any]: Métadonnées du document
        """
        # Générer un ID unique pour le document
        doc_id = str(uuid.uuid4())
        
        # Déterminer l'extension
        _, ext = os.path.splitext(filename)
        ext = ext.lower()
        
        # Créer le chemin de stockage
        file_path = os.path.join(self.storage_dir, f"{doc_id}{ext}")
        
        # Sauvegarder le fichier
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        # Extraire le texte selon le format
        if ext == '.txt' or ext == '.md':
            content = self.load_text(file_path)
        elif ext == '.pdf':
            content = self.load_pdf(file_path)
        else:
            raise ValueError(f"Format de fichier non supporté: {ext}")
        
        # Retourner les métadonnées
        return {
            "id": doc_id,
            "filename": filename,
            "path": file_path,
            "content": content,
            "extension": ext
        }