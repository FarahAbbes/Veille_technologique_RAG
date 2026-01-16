# Guide d'Installation et Configuration

## Prérequis
- Windows avec Python 3.11 installé
- `py` launcher disponible

## Environnement
- Créer l'environnement:
  - `py -3.11 -m venv .venv`
- Activer:
  - `.\.venv\Scripts\Activate.ps1`
- Vérifier:
  - `python --version`

## Dépendances
- Installer:
  - `python -m pip install -r requirements.txt`

## Composants optionnels
- Tesseract OCR:
  - Installer Tesseract Windows
  - Ajouter `tesseract.exe` au PATH
- Java pour Tabula:
  - Installer Java JRE
  - Vérifier `java -version`
- Ghostscript pour Camelot:
  - Installer Ghostscript

## Lancer l'application
- Basique:
  - `streamlit run app.py`
- Avancée:
  - `streamlit run app_advanced.py`

## Dépannage
- Erreur torch.classes:
  - Fichier `.streamlit/config.toml` avec `fileWatcherType = "none"`
- Modules manquants:
  - Réactiver `.venv`, réinstaller requirements
- OCR sans résultat:
  - Vérifier installation Tesseract et PATH

## Conseils
- Utiliser l'app avancée pour RRF, reranking et cache TTL
- Ajuster `k`, `fetch_k` et `TTL` selon la taille des documents
