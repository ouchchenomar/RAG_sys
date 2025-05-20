// Variables globales
const API_URL = window.location.protocol + "//" + window.location.host + "/api";
let systemInfo = null;

// Fonction pour charger les informations système
async function loadSystemInfo() {
    try {
        const response = await fetch(`${API_URL}/system/`);
        if (!response.ok) {
            throw new Error(`Erreur HTTP: ${response.status}`);
        }
        
        systemInfo = await response.json();
        displaySystemInfo(systemInfo);
    } catch (error) {
        console.error("Erreur lors du chargement des informations système:", error);
        document.getElementById("system-info").innerHTML = `
            <div class="alert alert-danger">
                Erreur de connexion au serveur. Veuillez vérifier que l'API est en cours d'exécution.
            </div>
        `;
    }
    
    // Charger également les informations de la Banque Mondiale
    loadWorldBankInfo();
}

// Fonction pour charger les informations système de la Banque Mondiale
async function loadWorldBankInfo() {
    try {
        const response = await fetch(`${API_URL}/worldbank/system-info/`);
        if (!response.ok) {
            throw new Error(`Erreur HTTP: ${response.status}`);
        }
        
        const info = await response.json();
        displayWorldBankInfo(info);
    } catch (error) {
        console.error("Erreur lors du chargement des informations système de la Banque Mondiale:", error);
        document.getElementById("worldbank-info").innerHTML = `
            <div class="alert alert-danger">
                Erreur de connexion au serveur. Veuillez vérifier que l'API est en cours d'exécution.
            </div>
        `;
    }
}

// Fonction pour afficher les informations de la Banque Mondiale
function displayWorldBankInfo(info) {
    const infoElement = document.getElementById("worldbank-info");
    
    if (!infoElement) {
        console.warn("Élément #worldbank-info non trouvé dans le DOM");
        return;
    }
    
    if (info.success) {
        infoElement.innerHTML = `
            <div class="card mb-4">
                <div class="card-header bg-primary text-white">
                    Statut du système RAG pour la Banque Mondiale
                </div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-3">
                            <div class="card text-center">
                                <div class="card-body">
                                    <h5 class="card-title">API</h5>
                                    <p class="card-text">${info.api_status}</p>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="card text-center">
                                <div class="card-body">
                                    <h5 class="card-title">Serveur</h5>
                                    <p class="card-text">${info.server_status}</p>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="card text-center">
                                <div class="card-body">
                                    <h5 class="card-title">Documents</h5>
                                    <p class="card-text">${info.document_count}</p>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="card text-center">
                                <div class="card-body">
                                    <h5 class="card-title">Chunks</h5>
                                    <p class="card-text">${info.chunk_count}</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    } else {
        infoElement.innerHTML = `
            <div class="alert alert-warning">
                ${info.message || "Impossible de récupérer les informations système."}
            </div>
        `;
    }
}

// Fonction pour mettre à jour la base de connaissances de la Banque Mondiale
async function updateWorldBankKnowledge(countries, indicators, includeTopics) {
    try {
        // Afficher un message de chargement
        document.getElementById("update-status").innerHTML = `
            <div class="alert alert-info">
                <div class="loader"></div> Mise à jour de la base de connaissances en cours...
            </div>
        `;
        
        const response = await fetch(`${API_URL}/worldbank/update-knowledge/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                countries: countries,
                indicators: indicators,
                include_topics: includeTopics
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `Erreur HTTP: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.success) {
            document.getElementById("update-status").innerHTML = `
                <div class="alert alert-success">
                    ${result.message || "Base de connaissances mise à jour avec succès"}
                </div>
            `;
            
            // Recharger les informations système
            loadWorldBankInfo();
        } else {
            document.getElementById("update-status").innerHTML = `
                <div class="alert alert-danger">
                    Erreur: ${result.message || "Une erreur est survenue"}
                </div>
            `;
        }
        
        return result;
    } catch (error) {
        console.error("Erreur lors de la mise à jour de la base de connaissances:", error);
        document.getElementById("update-status").innerHTML = `
            <div class="alert alert-danger">
                Erreur: ${error.message || "Erreur de connexion au serveur"}
            </div>
        `;
        
        return { 
            success: false, 
            message: error.message || "Erreur de connexion au serveur" 
        };
    }
}

// Fonction pour envoyer une requête à la Banque Mondiale
async function queryWorldBank(question) {
    try {
        // Ajouter la question au chat
        addMessageToChat(question, 'user');
        
        // Afficher un message de chargement
        const chatHistory = document.getElementById("chat-history");
        const loadingElement = document.createElement("div");
        loadingElement.className = "system-message";
        loadingElement.innerHTML = `<div class="loader"></div> Traitement de votre question...`;
        chatHistory.appendChild(loadingElement);
        chatHistory.scrollTop = chatHistory.scrollHeight;
        
        const response = await fetch(`${API_URL}/worldbank/query/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                question: question,
                top_k: 3,
                max_tokens: 512
            })
        });
        
        // Supprimer le message de chargement
        chatHistory.removeChild(loadingElement);
        
        if (!response.ok) {
            const errorData = await response.json();
            const errorMessage = errorData.detail || `Erreur HTTP: ${response.status}`;
            addMessageToChat(`Erreur: ${errorMessage}`, 'system');
            throw new Error(errorMessage);
        }
        
        const result = await response.json();
        
        if (result.success) {
            addMessageToChat(result, 'assistant');
        } else {
            addMessageToChat(`Erreur: ${result.message || 'Une erreur est survenue'}`, 'system');
        }
        
        return result;
    } catch (error) {
        console.error("Erreur lors de l'envoi de la requête:", error);
        return { 
            success: false, 
            message: error.message || "Erreur de connexion au serveur" 
        };
    }
}

// Fonction pour afficher les informations système
function displaySystemInfo(info) {
    const systemInfoElement = document.getElementById("system-info");
    
    if (info.success) {
        systemInfoElement.innerHTML = `
            <div class="row">
                <div class="col-md-3">
                    <div class="card text-center">
                        <div class="card-body">
                            <h5 class="card-title">${info.document_count}</h5>
                            <p class="card-text">Documents</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card text-center">
                        <div class="card-body">
                            <h5 class="card-title">${info.chunk_count}</h5>
                            <p class="card-text">Chunks</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card text-center">
                        <div class="card-body">
                            <h5 class="card-title">${info.model_loaded ? 'Actif' : 'Inactif'}</h5>
                            <p class="card-text">Modèle: ${info.model_name || 'Non chargé'}</p>
                        </div>
                    </div>
                </div>
            </div>
        `;
    } else {
        systemInfoElement.innerHTML = `
            <div class="alert alert-warning">
                Impossible de récupérer les informations système.
            </div>
        `;
    }
}

// Fonction pour télécharger un document
async function uploadDocument(formData) {
    try {
        const response = await fetch(`${API_URL}/documents/`, {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        return result;
    } catch (error) {
        console.error("Erreur lors du téléchargement du document:", error);
        return { 
            success: false, 
            message: "Erreur de connexion au serveur" 
        };
    }
}

// Fonction pour envoyer une requête
async function sendQuery(question) {
    try {
        const response = await fetch(`${API_URL}/query/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                question: question,
                top_k: 3,
                max_tokens: 512
            })
        });
        
        const result = await response.json();
        return result;
    } catch (error) {
        console.error("Erreur lors de l'envoi de la requête:", error);
        return { 
            success: false, 
            message: "Erreur de connexion au serveur" 
        };
    }
}

// Fonction pour ajouter un message au chat
function addMessageToChat(message, type) {
    const chatHistory = document.getElementById("chat-history");
    const messageElement = document.createElement("div");
    messageElement.className = `${type}-message`;
    
    if (type === 'assistant' && message.sources) {
        messageElement.innerHTML = `
            <p>${message.answer}</p>
            <div class="sources-info">
                <strong>Sources:</strong>
                <ul>
                    ${message.sources.map((source, index) => 
                        `<li>${source.filename} (Score: ${source.score.toFixed(4)})</li>`
                    ).join('')}
                </ul>
            </div>
        `;
    } else {
        messageElement.innerHTML = `<p>${message}</p>`;
    }
    
    chatHistory.appendChild(messageElement);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

// Fonction pour charger la liste des documents
async function loadDocumentList() {
    try {
        const response = await fetch(`${API_URL}/documents/`);
        if (!response.ok) {
            throw new Error(`Erreur HTTP: ${response.status}`);
        }
        
        const documents = await response.json();
        displayDocuments(documents);
    } catch (error) {
        console.error("Erreur lors du chargement des documents:", error);
        document.getElementById("document-list").innerHTML = `
            <li class="list-group-item text-danger">
                Impossible de charger la liste des documents.
            </li>
        `;
    }
}

// Fonction pour afficher la liste des documents
function displayDocuments(documents) {
    const documentList = document.getElementById("document-list");
    
    if (documents.length === 0) {
        documentList.innerHTML = `
            <li class="list-group-item">
                Aucun document disponible. Téléchargez-en un pour commencer.
            </li>
        `;
        return;
    }
    
    documentList.innerHTML = documents.map(doc => `
        <li class="list-group-item d-flex justify-content-between align-items-center">
            ${doc.filename}
            <span class="badge bg-primary rounded-pill">${doc.chunk_count} chunks</span>
        </li>
    `).join('');
}

// Fonction pour rafraîchir les données
function refreshData() {
    loadSystemInfo();
    loadDocumentList();
}

// Événements du document
document.addEventListener('DOMContentLoaded', () => {
    // Charger les informations système et la liste des documents
    refreshData();
    
    // Gérer le formulaire d'upload
    const uploadForm = document.getElementById('upload-form');
    if (uploadForm) {
        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const fileInput = document.getElementById('document');
            if (!fileInput.files[0]) {
                alert("Veuillez sélectionner un fichier");
                return;
            }
            
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            
            const uploadStatus = document.getElementById('upload-status');
            uploadStatus.innerHTML = `
                <div class="alert alert-info">
                    <div class="loader"></div> Téléchargement en cours...
                </div>
            `;
            
            const result = await uploadDocument(formData);
            
            if (result.success) {
                uploadStatus.innerHTML = `
                    <div class="alert alert-success">
                        Document téléchargé avec succès. ID: ${result.doc_id}
                    </div>
                `;
                
                // Recharger les informations système et la liste des documents
                refreshData();
                
                // Réinitialiser le formulaire
                uploadForm.reset();
            } else {
                uploadStatus.innerHTML = `
                    <div class="alert alert-danger">
                        Erreur: ${result.message}
                    </div>
                `;
            }
        });
    }
    
    // Gérer le formulaire de requête du RAG général
    const queryForm = document.getElementById('query-form');
    if (queryForm) {
        queryForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const queryInput = document.getElementById('query-input');
            const question = queryInput.value.trim();
            
            if (!question) return;
            
            // Ajouter la question au chat
            addMessageToChat(question, 'user');
            
            // Afficher un message de chargement
            const chatHistory = document.getElementById("chat-history");
            const loadingElement = document.createElement("div");
            loadingElement.className = "system-message";
            loadingElement.innerHTML = `<div class="loader"></div> Traitement de votre question...`;
            chatHistory.appendChild(loadingElement);
            chatHistory.scrollTop = chatHistory.scrollHeight;
            
            // Envoyer la requête
            const result = await sendQuery(question);
            
            // Supprimer le message de chargement
            chatHistory.removeChild(loadingElement);
            
            if (result.success) {
                addMessageToChat(result, 'assistant');
            } else {
                addMessageToChat(`Erreur: ${result.message || 'Une erreur est survenue'}`, 'system');
            }
            
            // Réinitialiser le formulaire
            queryInput.value = '';
        });
    }
    
    // Gérer le formulaire pour la mise à jour de la base de connaissances de la Banque Mondiale
    const worldbankForm = document.getElementById('worldbank-form');
    if (worldbankForm) {
        worldbankForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            // Récupérer les pays sélectionnés
            const selectedCountries = Array.from(document.querySelectorAll('input[name="country"]:checked')).map(input => input.value);
            
            // Récupérer les indicateurs sélectionnés
            const selectedIndicators = Array.from(document.querySelectorAll('input[name="indicator"]:checked')).map(input => input.value);
            
            // Vérifier si les thèmes sont inclus
            const includeTopics = document.getElementById('include-topics').checked;
            
            if (selectedCountries.length === 0) {
                alert("Veuillez sélectionner au moins un pays");
                return;
            }
            
            if (selectedIndicators.length === 0) {
                alert("Veuillez sélectionner au moins un indicateur");
                return;
            }
            
            // Mettre à jour la base de connaissances
            const result = await updateWorldBankKnowledge(selectedCountries, selectedIndicators, includeTopics);
            
            // Le statut est déjà affiché dans la fonction updateWorldBankKnowledge
        });
    }
    
    // Gérer le formulaire de requête pour la Banque Mondiale
    const worldbankQueryForm = document.getElementById('worldbank-query-form');
    if (worldbankQueryForm) {
        worldbankQueryForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const queryInput = document.getElementById('worldbank-query-input');
            const question = queryInput.value.trim();
            
            if (!question) return;
            
            // Envoyer la requête à la Banque Mondiale
            await queryWorldBank(question);
            
            // Réinitialiser le formulaire
            queryInput.value = '';
        });
    }
});