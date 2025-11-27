#  Trajets Kinshasa - Visualisation Graphique

Une application Streamlit pour visualiser et analyser les itinéraires routiers à Kinshasa utilisant l'API Google Maps et des algorithmes de graphes.

## Fonctionnalités

- **Calcul d'itinéraires** via l'API Google Maps
- **Trois modes de visualisation** :
  - Carte géographique avec fond OpenStreetMap
  - Graphe orienté simple
  - Graphe détaillé avec algorithme de Dijkstra
- **Comparaison multiple** d'itinéraires alternatifs
- **Export des données** en format texte
- **Interface responsive** et moderne


## Installation

### Prérequis
- Python 3.8+
- Clé API Google Maps
###Dépannage
Erreur : REQUEST_DENIED - Quota dépassé
Si vous rencontrez cette erreur :
Erreur lors du calcul de l'itinéraire: Échec du géocodage: REQUEST_DENIED
Causes :
-Quota quotidien de l'API Google Maps dépassé

-Limite du forfait gratuit atteinte

Solution :
Veuillez réessayer dans 24 heures - les quotas Google se réinitialisent généralement toutes les 24 heures.


### Installation des dépendances

```bash
pip install -r requirements.txt

