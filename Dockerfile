# Dockerfile pour FastAPI + Streamlit

# 1. Image de base
FROM python:3.12-slim

# 2. Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Installer les dépendances
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copier le projet
COPY . .

# 5. Exposer les ports
# 8000 pour FastAPI, 8501 pour Streamlit
EXPOSE 8000
EXPOSE 8502

# 6. Commande par défaut : démarrer FastAPI et Streamlit
# On utilisera docker-compose pour gérer les deux services
CMD ["sleep", "infinity"]