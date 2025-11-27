import os
import streamlit as st
import requests
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import networkx as nx
from typing import Dict, List, Tuple
import polyline
import time
import contextily as ctx
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ----------------------------
# CONFIGURATION STREAMLIT + CSS
# ----------------------------

st.set_page_config(
    page_title="Trajets Kinshasa",
    layout="wide",
    page_icon="🗺️"
)

# Thème CSS moderne
st.markdown("""
    <style>
        /* GLOBAL */
        body {
            font-family: 'Segoe UI', sans-serif;
        }
        .main {
            padding: 2rem;
        }
        /* TITRE */
        h1, h2, h3 {
            font-weight: 700 !important;
        }
        /* CARTE PRINCIPALE */
        .block-container {
            padding-top: 2rem;
        }
        /* INPUT */
        .stTextInput > div > input {
            border-radius: 10px;
            border: 1px solid #cccccc;
            padding: 10px;
            font-size: 16px;
        }
        /* BOUTON */
        .stButton>button {
            background: linear-gradient(90deg, #0072ff, #00c6ff);
            color: white;
            padding: 10px 20px;
            border-radius: 12px;
            border: none;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #005fcc, #00aadd);
        }
        /* CARDS */
        .result-card {
            padding: 15px;
            background: #f5f8ff;
            border: 1px solid #d2e3fc;
            border-radius: 12px;
            margin-bottom: 12px;
        }
        .route-card {
            padding: 15px;
            background: #f0f8ff;
            border: 2px solid #4aa3ff;
            border-radius: 12px;
            margin-bottom: 15px;
        }
        .best-route {
            border-color: #FF6BA1FF;
            background: #fff0f0;
        }
        .download-btn {
            background: linear-gradient(90deg, #28a745, #20c997);
            color: white;
            padding: 8px 16px;
            border-radius: 8px;
            border: none;
            font-size: 14px;
            cursor: pointer;
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# SERVICE GOOGLE MAPS
# ----------------------------

class GoogleMapsService:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or (
            st.secrets.get("GOOGLE_MAPS_API_KEY")
            if "GOOGLE_MAPS_API_KEY" in st.secrets
            else os.environ.get("GOOGLE_MAPS_API_KEY")
        )
        self.geocode_url = "https://maps.googleapis.com/maps/api/geocode/json"
        self.directions_url = "https://maps.googleapis.com/maps/api/directions/json"
        
    def geocode(self, address: str) -> Tuple[float, float]:
        """Convertit une adresse en coordonnées géographiques"""
        if not address.strip():
            raise ValueError("L'adresse ne peut pas être vide")
            
        # Ajouter Kinshasa, RDC si ce n'est pas spécifié
        if "kinshasa" not in address.lower() and "rdc" not in address.lower():
            address = f"{address}, Kinshasa, RDC"
            
        params = {"address": address, "key": self.api_key}
        
        print(f"DEBUG: Géocodage de l'adresse: {address}")
        
        try:
            response = requests.get(self.geocode_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            status = data.get("status")
            
            if status != "OK":
                error_msg = f"Échec du géocodage: {status}"
                if status == "ZERO_RESULTS":
                    error_msg = f"Adresse introuvable: '{address}'. Vérifiez l'orthographe."
                raise Exception(error_msg)
                
            location = data["results"][0]["geometry"]["location"]
            lat, lng = location["lat"], location["lng"]
            print(f"DEBUG: Coordonnées trouvées: ({lat}, {lng})")
            return lat, lng
            
        except requests.RequestException as e:
            raise Exception(f"Erreur réseau: {str(e)}")
        
    def get_routes(self, start: Tuple[float, float], end: Tuple[float, float], 
                  mode: str = "driving") -> List[Dict]:
        """Récupère les itinéraires entre deux points"""
        params = {
            "origin": f"{start[0]},{start[1]}",
            "destination": f"{end[0]},{end[1]}",
            "mode": mode,
            "alternatives": "true",
            "key": self.api_key,
        }
        
        print(f"DEBUG: Calcul d'itinéraire de {start} à {end} en mode {mode}")
        
        try:
            response = requests.get(self.directions_url, params=params, timeout=25)
            response.raise_for_status()
            
            data = response.json()
            status = data.get("status")
            
            if status != "OK":
                error_msg = f"Échec du calcul d'itinéraire: {status}"
                if status == "ZERO_RESULTS":
                    error_msg = f"Aucun itinéraire trouvé entre ces points. Vérifiez que les adresses sont dans la même région."
                elif status == "NOT_FOUND":
                    error_msg = "Point de départ ou destination introuvable."
                elif status == "MAX_ROUTE_LENGTH_EXCEEDED":
                    error_msg = "Distance trop longue pour l'API Google Maps."
                raise Exception(error_msg)
                
            return self._parse_routes(data.get("routes", []))
            
        except requests.RequestException as e:
            raise Exception(f"Erreur réseau: {str(e)}")
        
    def _parse_routes(self, routes_data: List[Dict]) -> List[Dict]:
        """Parse les données d'itinéraires de l'API Google"""
        routes = []
        
        for i, route in enumerate(routes_data):
            leg = route["legs"][0]
            
            # Décodage du tracé
            points_str = route.get("overview_polyline", {}).get("points", "")
            coords = polyline.decode(points_str) if points_str else []
            
            # Conversion des coordonnées (lat,lon → lon,lat)
            coords = [(lon, lat) for lat, lon in coords]
            
            route_info = {
                "coords": coords,
                "distance_km": leg["distance"]["value"] / 1000,
                "duration_min": leg["duration"]["value"] / 60,
                "steps": self._extract_steps(leg.get("steps", [])),
                "index": i,
                "start_address": leg.get("start_address", ""),
                "end_address": leg.get("end_address", ""),
                "distance_text": leg["distance"]["text"],
                "duration_text": leg["duration"]["text"],
            }
            routes.append(route_info)
            
        # Identifier le meilleur itinéraire (le plus court)
        if routes:
            best_idx = min(range(len(routes)), key=lambda k: routes[k]["distance_km"])
            for i, route in enumerate(routes):
                route["is_best"] = (i == best_idx)
                
        return routes
        
    def _extract_steps(self, steps: List[Dict]) -> List[Dict]:
        """Extrait les informations des étapes de l'itinéraire"""
        extracted = []
        for step in steps:
            extracted.append({
                "name": self._clean_html(step.get("html_instructions", "")),
                "distance": step["distance"]["value"],
                "distance_text": step["distance"]["text"],
                "duration_text": step["duration"]["text"],
            })
        return extracted
        
    def _clean_html(self, text: str) -> str:
        """Nettoie le HTML des instructions"""
        import re
        return re.sub(r"<[^>]+>", "", text).strip()

# ----------------------------
# WIDGET CARTE (MapWidget)
# ----------------------------

class MapWidget:
    def __init__(self):
        self.fig = None
        self.ax = None
        self.zoom_scale = 1.0
        self.view_center = None
        self.has_basemap = False

    def _init_map(self):
        """Initialise la carte"""
        self.fig = Figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Carte des itinéraires - Kinshasa", fontsize=14, fontweight='bold')
        self.ax.set_xlabel("Longitude")
        self.ax.set_ylabel("Latitude")
        self.ax.grid(True, alpha=0.3)
        self.has_basemap = False

    def add_basemap(self, bounds):
        """
        Ajoute un fond de carte OpenStreetMap
        bounds = (min_lon, max_lon, min_lat, max_lat)
        """
        try:
            # Ajouter le fond de carte OpenStreetMap
            ctx.add_basemap(self.ax, crs='EPSG:4326', 
                           source=ctx.providers.OpenStreetMap.Mapnik)
            self.has_basemap = True
        except Exception as e:
            try:
                # Fallback: essayer avec un autre provider
                ctx.add_basemap(self.ax, crs='EPSG:4326',
                               source=ctx.providers.Stamen.TonerLite)
                self.has_basemap = True
            except Exception as e2:
                self.has_basemap = False

    def calculate_bounds(self, routes):
        """Calcule les limites de la carte basées sur tous les itinéraires"""
        all_lons = []
        all_lats = []
        
        for route in routes:
            coords = route.get("coords", [])
            if coords:
                lons, lats = zip(*coords)
                all_lons.extend(lons)
                all_lats.extend(lats)
        
        if not all_lons:
            # Coordonnées par défaut pour Kinshasa
            return (15.2, 15.4, -4.4, -4.2)
        
        min_lon, max_lon = min(all_lons), max(all_lons)
        min_lat, max_lat = min(all_lats), max(all_lats)
        
        # Ajouter une marge
        lon_margin = (max_lon - min_lon) * 0.1
        lat_margin = (max_lat - min_lat) * 0.1
        
        return (min_lon - lon_margin, max_lon + lon_margin, 
                min_lat - lat_margin, max_lat + lat_margin)

    def plot_routes(self, routes: List[Dict], route_colors: Dict[int, str] = None):
        """
        Affiche les itinéraires sur la carte et retourne la figure
        """
        self._init_map()

        if not routes:
            return self.fig

        # Couleurs par défaut si non fournies
        if route_colors is None:
            route_colors = {
                0: "#816bff",
                1: "#4ecdc4",
                2: "#d80bf7",
                3: "#eff315",
            }

        # Tracer chaque itinéraire
        for route in routes:
            coords = route.get("coords", [])
            if not coords:
                continue
            
            xs, ys = zip(*coords)
            color = route_colors.get(route.get("index", 0), "#666666")
            linewidth = 4 if route.get("is_best", False) else 2
            alpha = 1.0 if route.get("is_best", False) else 0.7
            linestyle = '-' if route.get("is_best", False) else '--'
            
            self.ax.plot(xs, ys, color=color, linewidth=linewidth, 
                        alpha=alpha, linestyle=linestyle, zorder=10,
                        label=f"{'' if route.get('is_best', False) else ''}Itinéraire {route.get('index', 0) + 1}")

        # Marqueurs de départ et arrivée
        best_route = next((r for r in routes if r.get("is_best", False)), routes[0])
        if best_route.get("coords"):
            start_lon, start_lat = best_route["coords"][0]
            end_lon, end_lat = best_route["coords"][-1]
            
            # Départ (vert)
            self.ax.scatter(start_lon, start_lat, c='green', s=200, 
                           edgecolors='white', linewidth=2, zorder=11,
                           marker='o', label='Départ')
            
            # Arrivée (rouge)
            self.ax.scatter(end_lon, end_lat, c='red', s=200,
                           edgecolors='white', linewidth=2, zorder=11,
                           marker='s', label='Arrivée')

        # Calculer et définir les limites
        bounds = self.calculate_bounds(routes)
        self.ax.set_xlim(bounds[0], bounds[1])
        self.ax.set_ylim(bounds[2], bounds[3])

        # Ajouter le fond de carte
        self.add_basemap(bounds)

        # Légende améliorée
        self.ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

        # Style amélioré
        self.ax.tick_params(axis='both', which='major', labelsize=10)
        
        return self.fig

    def display_route_info(self, routes: List[Dict]):
        """Affiche les informations des itinéraires sous forme de cartes"""
        if not routes:
            return

        st.subheader("Informations détaillées des itinéraires")
        
        for route in routes:
            is_best = route.get("is_best", False)
            border_color = "#6b7aff" if is_best else "#4aa3ff"
            background_color = "#fff0f0" if is_best else "#f5f8ff"
            
            st.markdown(f"""
            <div style="
                padding: 15px; 
                background: {background_color}; 
                border: 2px solid {border_color}; 
                border-radius: 12px; 
                margin-bottom: 15px;
            ">
                <h4 style="margin:0; color: {border_color};">
                    {' Itinéraire Optimal' if is_best else f'Itinéraire {route.get("index", 0) + 1}'}
                </h4>
                <b> Distance:</b> {route.get('distance_text', 'N/A')} ({route.get('distance_km', 0):.1f} km)<br>
                <b> Durée:</b> {route.get('duration_text', 'N/A')} ({route.get('duration_min', 0):.1f} min)<br>
                <b> Départ:</b> {route.get('start_address', 'N/A')}<br>
                <b> Arrivée:</b> {route.get('end_address', 'N/A')}<br>
                <b> Nombre d'étapes:</b> {len(route.get('steps', []))}
            </div>
            """, unsafe_allow_html=True)

            # Afficher les étapes détaillées pour le meilleur itinéraire
            if is_best and route.get('steps'):
                with st.expander("📋 Voir les étapes détaillées"):
                    for i, step in enumerate(route['steps']):
                        st.markdown(f"""
                        **Étape {i+1}:** {step['name']}  
                        *Distance: {step.get('distance_text', 'N/A')} - Durée: {step.get('duration_text', 'N/A')}*
                        """)

# ----------------------------
# GRAPH WIDGET (Dijkstra)
# ----------------------------

class GraphWidget:
    def __init__(self):
        self.fig = None
        self.ax = None
        self.graph = nx.DiGraph()
        self.dijkstra_path = None
        self.node_mapping = {}
        self.node_counter = 1

    def _init_graph(self):
        """Initialise le graphe"""
        self.fig = Figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Graphe des itinéraires (Algorithme de Dijkstra)", fontsize=14, fontweight='bold')
        self.ax.axis('off')
        self.node_mapping.clear()
        self.node_counter = 1

    def plot_graph(self, routes: List[Dict], route_colors: Dict[int, str] = None):
        """
        Affiche le graphe avec l'algorithme de Dijkstra et retourne la figure
        """
        self._init_graph()
        self.graph.clear()

        if not routes:
            self.ax.text(0.5, 0.5, "Aucun itinéraire à afficher",
                         ha='center', va='center', fontsize=12)
            return self.fig

        # Couleurs par défaut
        if route_colors is None:
            route_colors = {
                0: "#816bff",
                1: "#4ecdc4", 
                2: "#d80bf7",
                3: "#eff315",
            }

        # Construction du graphe
        self._build_graph_from_routes(routes)

        # Dessin du graphe
        self._draw_graph(route_colors)

        return self.fig

    def _build_graph_from_routes(self, routes: List[Dict]):
        """Construit le graphe à partir des itinéraires"""
        # Ajouter les nœuds de départ et d'arrivée
        self.graph.add_node("start", type="start", node_number="D")
        self.graph.add_node("end", type="end", node_number="A")

        for route in routes:
            route_id = f"route_{route['index']}"
            prev_node = "start"

            # Ajouter les étapes comme nœuds intermédiaires
            for i, step in enumerate(route.get('steps', [])):
                node_id = f"{route_id}_step_{i}"
                node_number = self.node_counter
                self.node_counter += 1

                self.graph.add_node(
                    node_id,
                    route_index=route['index'],
                    node_number=str(node_number),
                    label=step['name'],
                    distance=step['distance_text'],
                    duration=step['duration_text']
                )

                self.node_mapping[str(node_number)] = {
                    'label': step['name'],
                    'route_index': route['index'],
                    'step_index': i,
                    'distance': step['distance_text'],
                    'duration': step['duration_text']
                }

                # Ajouter l'arête avec le poids (distance)
                self.graph.add_edge(
                    prev_node, node_id,
                    weight=step['distance'],
                    route_index=route['index']
                )
                prev_node = node_id

            # Ajouter l'arête finale vers le nœud d'arrivée
            self.graph.add_edge(
                prev_node, "end",
                weight=1,  # Poids minimal pour la dernière étape
                route_index=route['index']
            )

        # Calculer le chemin optimal avec Dijkstra
        try:
            self.dijkstra_path = nx.shortest_path(self.graph, "start", "end", weight="weight")
        except nx.NetworkXNoPath:
            self.dijkstra_path = None

    def _draw_graph(self, route_colors: Dict[int, str]):
        """Dessine le graphe avec NetworkX"""
        if self.graph.number_of_nodes() == 0:
            return

        # Positionnement des nœuds
        pos = nx.spring_layout(self.graph, k=3, iterations=50, scale=2.0)

        # Catégoriser les nœuds
        start_nodes = [n for n in self.graph.nodes() if self.graph.nodes[n].get('type') == 'start']
        end_nodes = [n for n in self.graph.nodes() if self.graph.nodes[n].get('type') == 'end']
        route_nodes = [n for n in self.graph.nodes() if n not in start_nodes + end_nodes]

        # Dessiner les nœuds
        nx.draw_networkx_nodes(self.graph, pos, nodelist=start_nodes,
                               node_color='green', node_size=1200, ax=self.ax)
        nx.draw_networkx_nodes(self.graph, pos, nodelist=end_nodes,
                               node_color='red', node_size=1200, ax=self.ax)

        # Grouper les nœuds par itinéraire
        route_groups = {}
        for node in route_nodes:
            idx = self.graph.nodes[node].get('route_index')
            route_groups.setdefault(idx, []).append(node)

        for route_idx, nodes in route_groups.items():
            color = route_colors.get(route_idx, '#666666')
            nx.draw_networkx_nodes(self.graph, pos, nodelist=nodes,
                                   node_color=color, node_size=800, ax=self.ax)

        # Dessiner les arêtes avec couleurs et épaisseurs
        edge_colors = []
        edge_widths = []
        edge_styles = []

        for edge in self.graph.edges():
            route_idx = self.graph.edges[edge].get('route_index')
            color = route_colors.get(route_idx, '#666666')

            # Vérifier si cette arête fait partie du chemin optimal
            is_optimal = (
                self.dijkstra_path and
                edge in zip(self.dijkstra_path[:-1], self.dijkstra_path[1:])
            )
            
            width = 4.0 if is_optimal else 2.0
            style = '-' if is_optimal else '--'

            edge_colors.append(color)
            edge_widths.append(width)
            edge_styles.append(style)

        nx.draw_networkx_edges(self.graph, pos, edge_color=edge_colors,
                               width=edge_widths, style=edge_styles, 
                               arrows=True, ax=self.ax, arrowsize=20)

        # Labels des nœuds
        labels = {n: self.graph.nodes[n].get('node_number', '?')
                  for n in self.graph.nodes()}

        nx.draw_networkx_labels(self.graph, pos, labels, font_size=10,
                               font_weight='bold', ax=self.ax,
                               bbox=dict(boxstyle="circle,pad=0.3",
                                         facecolor="white",
                                         edgecolor='black',
                                         alpha=0.9))

        # Créer la légende
        self._create_legend(route_colors)

    def _create_legend(self, route_colors: Dict[int, str]):
        """Crée une légende pour le graphe"""
        legend_handles = []

        # Nœuds spéciaux
        legend_handles.append(Line2D([], [], marker='o', color='green',
                                     markersize=12, linestyle='None', label='Départ (D)'))
        legend_handles.append(Line2D([], [], marker='o', color='red',
                                     markersize=12, linestyle='None', label='Arrivée (A)'))

        # Itinéraires
        for route_idx, color in route_colors.items():
            legend_handles.append(
                Line2D([], [], color=color, linewidth=3,
                       label=f"Itinéraire {route_idx + 1}")
            )

        # Chemin optimal
        if self.dijkstra_path:
            legend_handles.append(
                Line2D([], [], color='black', linewidth=4, linestyle='-',
                       label='Chemin optimal (Dijkstra)')
            )

        self.ax.legend(handles=legend_handles,
                       loc='upper center',
                       bbox_to_anchor=(0.5, -0.05),
                       ncol=3,
                       fontsize=10,
                       frameon=True,
                       fancybox=True,
                       shadow=True)

    def export_node_details(self):
        """Exporte les détails des nœuds pour téléchargement"""
        content = "DÉTAILS DES ITINÉRAIRES - ALGORITHME DE DIJKSTRA\n"
        content += "=" * 50 + "\n\n"
        
        content += "POINTS SPÉCIAUX:\n"
        content += "D - Point de départ\n"
        content += "A - Point d'arrivée\n\n"
        
        content += "ÉTAPES PAR NUMÉRO:\n"
        for node_num, info in sorted(self.node_mapping.items(), key=lambda x: int(x[0])):
            content += f"{node_num}: {info['label']}\n"
            content += f"   Itinéraire {info['route_index'] + 1}, Étape {info['step_index'] + 1}\n"
            content += f"   Distance: {info['distance']}, Durée: {info['duration']}\n\n"
        
        return content

# ----------------------------
# FONCTIONS UTILITAIRES
# ----------------------------

def build_simple_graph_image(routes: List[Dict]):
    """Construit un graphe NetworkX simple et retourne une FIGURE matplotlib."""
    G = nx.DiGraph()

    # Ajouter les routes au graphe
    for route in routes:
        start = route.get("start_address", "Départ")
        end = route.get("end_address", "Arrivée")
        distance = route.get("distance_text", "")
        is_best = route.get("is_best", False)
        
        label = f" {distance}" if is_best else distance
        G.add_edge(start, end, label=label, is_best=is_best)

    # Positions simplifiées pour un rendu clair
    pos = nx.spring_layout(G, seed=42)

    fig = Figure(figsize=(12, 8))
    ax = fig.subplots()
    ax.axis('off')

    # Couleurs différentes pour la meilleure route
    edge_colors = []
    for u, v, data in G.edges(data=True):
        if data.get('is_best', False):
            edge_colors.append("#8364ff")
        else:
            edge_colors.append('#4aa3ff')

    # Nœuds
    nx.draw_networkx_nodes(G, pos, node_size=2000,
                           node_color="#4aa3ff", edgecolors="white", ax=ax)

    # Arêtes avec couleurs
    nx.draw_networkx_edges(G, pos, arrowstyle='->',
                           arrowsize=20, width=2, edge_color=edge_colors, ax=ax)

    # Labels des nœuds
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold",
                            font_color="white", ax=ax)

    # Labels des arêtes
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                 font_size=9, font_color="#333", ax=ax)

    return fig

# ----------------------------
# INTERFACE UTILISATEUR
# ----------------------------

st.title(" Trajets routiers à Kinshasa — Visualisation Graphique")

st.write("Entrez un point de départ et une destination pour obtenir les visualisations des trajets.")

# Initialisation du service
maps_service = GoogleMapsService()

col1, col2 = st.columns(2)

with col1:
    start_place = st.text_input("Point de départ (ex: Rond-Point Victoire)", "")
with col2:
    end_place = st.text_input("Destination (ex: Gare Centrale)", "")

col3, col4 = st.columns(2)
with col3:
    travel_mode = st.selectbox("Mode de transport", ["voiture", "A pied", "Vélo"], index=0)
with col4:
    st.markdown("<br>", unsafe_allow_html=True)
    launch = st.button(" Générer les visualisations")

# ----------------------------
# LOGIQUE PRINCIPALE
# ----------------------------

if launch:
    if not start_place or not end_place:
        st.error("Veuillez remplir les deux champs.")
        st.stop()

    # Afficher un indicateur de chargement
    with st.spinner("Calcul des itinéraires..."):
        try:
            # Géocodage des adresses
            start_coords = maps_service.geocode(start_place)
            end_coords = maps_service.geocode(end_place)

            # Récupération des itinéraires
            routes = maps_service.get_routes(start_coords, end_coords, mode=travel_mode)

            # Créer les instances des widgets
            map_widget = MapWidget()
            graph_widget = GraphWidget()
            
            # Définir les couleurs des itinéraires
            route_colors = {
                0: "#816bff",
                1: "#4ecdc4",
                2: "#d80bf7", 
                3: "#eff315",
            }
            
            # Afficher les informations des itinéraires
            map_widget.display_route_info(routes)
            
            # Créer des onglets pour les différentes visualisations
            tab1, tab2, tab3 = st.tabs([" Carte Géographique", " Graphe Simple", " Graphe Dijkstra"])
            
            with tab1:
                st.subheader(" Carte des itinéraires")
                fig_map = map_widget.plot_routes(routes, route_colors)
                st.pyplot(fig_map)
                
            with tab2:
                st.subheader("Graphe orienté simple")
                fig_simple_graph = build_simple_graph_image(routes)
                st.pyplot(fig_simple_graph, clear_figure=True)
                
            with tab3:
                st.subheader(" Graphe détaillé avec algorithme de Dijkstra")
                fig_dijkstra = graph_widget.plot_graph(routes, route_colors)
                st.pyplot(fig_dijkstra, clear_figure=True)
                
                # Bouton de téléchargement des détails
                details_content = graph_widget.export_node_details()
                st.download_button(
                    label=" Télécharger les détails de l'itinéraire",
                    data=details_content,
                    file_name="details_itineraire.txt",
                    mime="text/plain",
                    help="Téléchargez les détails complets de tous les itinéraires"
                )

        except Exception as e:
            st.error(f"Erreur lors du calcul de l'itinéraire: {str(e)}")