# 🚀 Guide d'Installation et d'Exécution

## 📋 Prérequis

- Python 3.10 ou supérieur
- pip (gestionnaire de paquets Python)
- Une clé API Google Gemini (optionnelle, pour utiliser Gemini)

---

## 🔧 Installation

### Étape 1 : Vérifier Python

Ouvrez un terminal (PowerShell ou CMD) et vérifiez que Python est installé :

```bash
python --version
```

Vous devriez voir quelque chose comme `Python 3.10.x` ou supérieur.

### Étape 2 : Naviguer vers le dossier du projet

```bash
cd C:\Users\lenovo\pdf-rag
```

### Étape 3 : Créer un environnement virtuel (recommandé)

```bash
python -m venv venv
```

### Étape 4 : Activer l'environnement virtuel

**Sur Windows (PowerShell) :**
```bash
.\venv\Scripts\Activate.ps1
```

**Sur Windows (CMD) :**
```bash
venv\Scripts\activate.bat
```

Vous devriez voir `(venv)` au début de votre ligne de commande.

### Étape 5 : Installer les dépendances

```bash
pip install -r requirements.txt
```

Cette étape peut prendre quelques minutes car elle télécharge plusieurs packages (PyTorch, LangChain, etc.).

---

## ▶️ Exécution

### Application de base (`app.py`)

```bash
streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur à l'adresse : `http://localhost:8501`

### Application avancée (`app_advanced.py`)

```bash
streamlit run app_advanced.py
```

L'application s'ouvrira automatiquement dans votre navigateur à l'adresse : `http://localhost:8501`

---

## 🔑 Configuration de la clé API Gemini

1. **Obtenir une clé API** :
   - Visitez [Google AI Studio](https://ai.google.dev/)
   - Créez un compte ou connectez-vous
   - Générez une nouvelle clé API

2. **Utiliser la clé dans l'application** :
   - Dans la barre latérale de l'application
   - Sélectionnez "Gemini" dans le menu déroulant "Modèle IA"
   - Entrez votre clé API dans le champ "Clé API Gemini"
   - La clé est déjà pré-remplie avec : `AIzaSyCM78aSjZCHiEH5uxehA5f9ru2xL2mHNcQ`

---

## 📝 Utilisation

### Application de base

1. **Télécharger des PDFs** :
   - Utilisez la barre latérale pour télécharger un ou plusieurs fichiers PDF
   - Cliquez sur "🚀 Analyser les PDFs"

2. **Poser des questions** :
   - Tapez votre question dans le champ de texte
   - L'application cherchera les réponses dans vos documents PDF

### Application avancée

1. **Configurer le système** :
   - Choisissez votre modèle (FLAN-T5 ou Gemini)
   - Ajustez les paramètres (TTL du cache, reranking, fusion RRF)
   - Configurez k et fetch_k selon vos besoins

2. **Construire l'index** :
   - Téléchargez vos PDFs
   - Cliquez sur "Construire/assurer l'index"

3. **Poser des questions** :
   - Tapez votre question
   - Le système utilise le cache, le reranking et la fusion RRF pour une meilleure réponse

---

## ⚠️ Dépannage

### Erreur : "Module not found"

Si vous obtenez une erreur `ModuleNotFoundError`, réinstallez les dépendances :

```bash
pip install --upgrade -r requirements.txt
```

### Erreur : "Port already in use"

Si le port 8501 est déjà utilisé, utilisez un autre port :

```bash
streamlit run app.py --server.port 8502
```

### Erreur : Gemini API not found

Si vous obtenez une erreur avec Gemini :
- Vérifiez que votre clé API est valide
- Vérifiez que votre région est supportée par l'API Gemini
- Assurez-vous que l'API Generative Language est activée dans Google Cloud Console

### Erreur : "torch not found"

Si PyTorch n'est pas installé correctement :

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## 🛑 Arrêter l'application

Pour arrêter l'application, appuyez sur `Ctrl + C` dans le terminal.

---

## 📦 Packages principaux installés

- **Streamlit** : Interface web
- **LangChain** : Gestion des LLM et chaînes
- **FAISS** : Base de données vectorielle
- **PyPDF2** : Extraction de texte des PDFs
- **Transformers** : Modèles HuggingFace (FLAN-T5)
- **Google Generative AI** : Support Gemini
- **Sentence Transformers** : Embeddings

---

## 💡 Conseils

- **Première utilisation** : Commencez par `app.py` qui est plus simple
- **Performance** : Utilisez `app_advanced.py` pour de meilleures performances avec cache et reranking
- **Modèles** : FLAN-T5 fonctionne localement, Gemini nécessite une connexion Internet et une clé API
- **PDFs** : Les PDFs avec du texte (pas seulement des images) fonctionnent mieux

---

## 🆘 Besoin d'aide ?

Si vous rencontrez des problèmes :
1. Vérifiez que toutes les dépendances sont installées
2. Assurez-vous d'utiliser Python 3.10+
3. Vérifiez que votre environnement virtuel est activé
4. Consultez les messages d'erreur pour plus de détails

---

**Bon développement ! 🚀**
