import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from scholarly import scholarly
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk,  UnidentifiedImageError
from io import BytesIO

# Función para obtener características de un artículo
def load_image_from_url(url, max_size=(100, 100)):
    try:
        response = requests.get(url)
        response.raise_for_status()
        image_data = BytesIO(response.content)
        image = Image.open(image_data)
        image.thumbnail(max_size)  # Redimensionar la imagen
        return ImageTk.PhotoImage(image)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def get_article_features(article):
    features = {
        'title': article['bib'].get('title', 'No title available'),
        'authors': article['bib'].get('author', 'No authors available'),
        'year': article['bib'].get('pub_year', 0),
        'source': 'Scholary',
        'num_citations': article.get('num_citations', 0),
        'url': article.get('eprint_url', 'No URL available')
    }
    print(features['title'])
    return features

def get_article_features_scopus(article):
    features = {
        'title': article.get('dc:title', 'No title'),
        'authors': article.get('dc:creator', 'No authors'),
        'year': article.get('prism:coverDate', 'No year available').split('-')[0],
        'source': 'Scopus',
        'num_citations': article.get('citedby-count', 0),
        'url': article['link'][2]['@href'] if 'link' in article and len(article['link']) > 2 else 'No URL available' 
    }
    print(features['title'])
    return features

def get_article_features_wos(article):
    features = {
        'title': article['title'],
        'authors': ', '.join(author['displayName'] for author in article.get('names', {}).get('authors', [{'displayName': 'No authors'}])),
        'year': article.get('source', {}).get('publishYear', 0),
        'source': 'Web Of Science',
        'num_citations': len(article.get('citations', [])),
        'url': article.get('links', {}).get('record', 'No URL available')
    }
    print(features['title'])
    return features


def search_scopus(query, api_key,start_index):
    url = "https://api.elsevier.com/content/search/scopus"
    headers = {
        "Accept": "application/json",
        "X-ELS-APIKey": api_key
    }
    params = {
        "query": query,
        "count": 20 , # Número de resultados por página
        "start" : start_index  # Índice desde el cual empezar
    }
    
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return None

def search_wos(query, api_key,i):
    url = "https://api.clarivate.com/apis/wos-starter/v1/documents"
    headers = {
        "X-APIKey": api_key
    }
    params = {
        "db":"WOS",
        "q": f'TS=({query})',
        "limit": 50,  # Número de resultados por página
        "page": i
    }
    
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.content}")
        return None

# Ejemplo de uso
api_key_scopus = "fd6fc76f2c794da24ea395504ad8d937"
api_key_wos = "5cf0e2bcdb141300c58ef24eaa76e786018e68dc"

def search_articles():
    query = entry.get()

    articles = []
    articleScopus = []
    articleWos= []
    
    i =0
    while i<80:
        results = search_scopus(query, api_key_scopus,i)
        if results:
            for article in results['search-results']['entry']:
                features = get_article_features_scopus(article)
                if features['url'] != 'No URL available':
                    articleScopus.append(features)
        i=i+19
    
    i =0
    while i<2:
        results = search_wos(query, api_key_wos,i)
        if results:
            for article in results.get('hits', []):
                features = get_article_features_wos(article)
                if features['url'] != 'No URL available':
                        articleWos.append(features)
        i=i+1
    
    i =0
    search_query = scholarly.search_pubs(query)
    for i, article in enumerate(search_query):
        features = get_article_features(article)
        if features['url'] != 'No URL available':
            articles.append(features)
        if len(articles) >= 50:  # Limite de artículos
            break
        

    if articles or articleWos or articleScopus:
        if articles:
            df_articles = pd.DataFrame(articles)
            df_articles['year'] = pd.to_numeric(df_articles['year'], errors='coerce')
            df_articles['num_citations'] = pd.to_numeric(df_articles['num_citations'], errors='coerce')
            df_articles.dropna(subset=['year', 'num_citations'], inplace=True)
            df_articles['year'] = df_articles['year'].astype(int)
            latest_year = df_articles['year'].max()
            most_citations = df_articles['num_citations'].max()
            reference_features = np.array([latest_year, most_citations])
            
            distances = []
            for index, row in df_articles.iterrows():
                feature_values = row[['year', 'num_citations']].values
                distance = np.linalg.norm(feature_values - reference_features)
                distances.append(distance)
                
            df_articles['distance'] = distances
            df_articles = df_articles.sort_values(by=['distance'], ascending=True)
            
        if articleWos:
            df_articles_wos = pd.DataFrame(articleWos)
            df_articles_wos['year'] = pd.to_numeric(df_articles_wos['year'], errors='coerce')
            df_articles_wos['num_citations'] = pd.to_numeric(df_articles_wos['num_citations'], errors='coerce')
            df_articles_wos.dropna(subset=['year', 'num_citations'], inplace=True)
            df_articles_wos['year'] = df_articles_wos['year'].astype(int)
            latest_year_wos = df_articles_wos['year'].max()
            most_citations_wos = df_articles_wos['num_citations'].max()
            reference_features_wos = np.array([latest_year_wos, most_citations_wos])
            
            distances_wos = []
            for index, row in df_articles_wos.iterrows():
                feature_values = row[['year', 'num_citations']].values
                distance = np.linalg.norm(feature_values - reference_features_wos)
                distances_wos.append(distance)
            
            df_articles_wos['distance'] = distances_wos
            df_articles_wos = df_articles_wos.sort_values(by=['distance'], ascending=True) 
            
        if articleScopus:
            df_articles_scopus = pd.DataFrame(articleScopus)
            df_articles_scopus['year'] = pd.to_numeric(df_articles_scopus['year'], errors='coerce')
            df_articles_scopus['num_citations'] = pd.to_numeric(df_articles_scopus['num_citations'], errors='coerce')
            df_articles_scopus.dropna(subset=['year', 'num_citations'], inplace=True)
            df_articles_scopus['year'] = df_articles_scopus['year'].astype(int)
            latest_year_scopus = df_articles_scopus['year'].max()
            most_citations_scopus = df_articles_scopus['num_citations'].max()
            reference_features_scopus = np.array([latest_year_scopus, most_citations_scopus])
            
            distances_scopus = []
            for index, row in df_articles_scopus.iterrows():
                feature_values = row[['year', 'num_citations']].values
                distance = np.linalg.norm(feature_values - reference_features_scopus)
                distances_scopus.append(distance)
                
            df_articles_scopus['distance'] = distances_scopus
            df_articles_scopus = df_articles_scopus.sort_values(by=['distance'], ascending=True)

        
        # Obtener recomendaciones
        # recommended_articles = df_articles.head(20)
        dataframes = [df for df in [df_articles.head(20), df_articles_wos.head(20), df_articles_scopus.head(20)] if not df.empty]

        # Concatenar solo los DataFrames no vacíos
        if dataframes:
            recommended_articles = pd.concat(dataframes, ignore_index=True)
        else:
            recommended_articles = pd.DataFrame()  # DataFrame vacío si todos los DataFrames están vacíos
        # Mostrar resultados en el Treeview
        for row in tree.get_children():
            tree.delete(row)
        for row in recommended_articles.itertuples(index=False):
            tree.insert('', tk.END, values=row)

        # Crear gráficos usando matplotlib y mostrarlos en Tkinter
        fig_dist.clf()
        ax1 = fig_dist.add_subplot(111)
        ax1.hist(df_articles['distance'], bins=30, edgecolor='k', alpha=0.7)
        ax1.set_title('Distribución de Distancias entre artículos evaluados')
        ax1.set_xlabel('Distancia')
        ax1.set_ylabel('Frecuencia')
        canvas_dist.draw()

        fig_comp.clf()
        ax2 = fig_comp.add_subplot(111)
        features_to_compare = ['year', 'num_citations']
        reference_values = reference_features
        recommended_values = recommended_articles[features_to_compare].mean().values

        x = np.arange(len(features_to_compare))  # El número de características
        width = 0.35  # El ancho de las barras

        rects1 = ax2.bar(x - width / 2, reference_values, width, label='Artículo de Referencia')
        rects2 = ax2.bar(x + width / 2, recommended_values, width, label='Artículos Recomendados')

        # Añadir etiquetas, título y leyenda
        ax2.set_ylabel('Valor')
        ax2.set_title('Comparación de Características entre los Artículos')
        ax2.set_xticks(x)
        ax2.set_xticklabels(features_to_compare)
        ax2.legend()

        canvas_comp.draw()
    else:
        print("No se encontraron artículos para la consulta proporcionada.")

# Crear la ventana principal de Tkinter
# root = tk.Tk()
# root.title("Sistema de recomendación de articulos")

# # Crear un marco para la entrada y el botón de búsqueda
# frame_input = ttk.Frame(root, padding="10")
# frame_input.grid(row=0, column=0, sticky=(tk.W, tk.E))

# entry = ttk.Entry(frame_input, width=50)
# entry.grid(row=0, column=0, padx=5)

# button = ttk.Button(frame_input, text="Buscar", command=search_articles)
# button.grid(row=0, column=1, padx=5)

# # Crear un marco para mostrar los resultados
# frame_results = ttk.Frame(root, padding="10")
# frame_results.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# tree = ttk.Treeview(frame_results, columns=['title', 'authors', 'year', 'source', 'num_citations', 'url', 'distance'], show='headings')
# for col in ['title', 'authors', 'year', 'source', 'num_citations', 'url', 'distance']:
#     tree.heading(col, text=col)
#     tree.column(col, width=150)
# tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# scrollbar = ttk.Scrollbar(frame_results, orient=tk.VERTICAL, command=tree.yview)
# tree.configure(yscroll=scrollbar.set)
# scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

# frame_results.columnconfigure(0, weight=1)
# frame_results.rowconfigure(0, weight=1)

# # Crear gráficos usando matplotlib y mostrarlos en Tkinter
# fig_dist = plt.figure(figsize=(8, 4))
# canvas_dist = FigureCanvasTkAgg(fig_dist, master=root)
# canvas_dist.get_tk_widget().grid(row=2, column=0, pady=5)

# fig_comp = plt.figure(figsize=(8, 4))
# canvas_comp = FigureCanvasTkAgg(fig_comp, master=root)
# canvas_comp.get_tk_widget().grid(row=3, column=0, pady=5)

# root.mainloop()

#------------------------------

# Crear la ventana principal de Tkinter
root = tk.Tk()
root.title("SISTEMA DE RECOMENDACIÓN                                                                                Luis Ortega, Adolfo Botero, Daniel Valencia")

# Crear un marco para la entrada y el botón de búsqueda
frame_input = ttk.Frame(root, padding="0")
frame_input.grid(row=0, column=0,sticky=(tk.W, tk.E))

entry = ttk.Entry(frame_input, width=50)
entry.grid(row=0, column=1, padx=1)

button = ttk.Button(frame_input, text="Buscar", command=search_articles)
button.grid(row=0, column=2, padx=3)
# Descargar la imagen desde la URL
url = "https://i0.wp.com/www.ucaldas.edu.co/portal/wp-content/uploads/2020/05/monitorias.jpg?w=700&ssl=1"  # Reemplaza esta URL con la URL de tu imagen
photo = load_image_from_url(url,max_size=(150, 400))

# Crear un marco para el logo
frame_logo = ttk.Frame(frame_input, padding="0")
frame_logo.grid(row=0, column=0, sticky=(tk.W, tk.E))

# Si se descargó y cargó la imagen correctamente, mostrarla
if photo:
    logo_label = tk.Label(frame_logo, image=photo)
    logo_label.image = photo  # Guardar una referencia para evitar que la imagen sea recolectada por el garbage collector
    logo_label.grid(row=0, column=0, pady=0, padx=10)


# Crear un marco para mostrar los resultados
frame_results = ttk.Frame(root, padding="10")
frame_results.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

tree = ttk.Treeview(frame_results, columns=['title', 'authors', 'year', 'source', 'num_citations', 'url', 'distance'], show='headings')
for col in ['title', 'authors', 'year', 'source', 'num_citations', 'url', 'distance']:
    tree.heading(col, text=col)
    tree.column(col, width=150)
tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

scrollbar = ttk.Scrollbar(frame_results, orient=tk.VERTICAL, command=tree.yview)
tree.configure(yscroll=scrollbar.set)
scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

frame_results.columnconfigure(0, weight=1)
frame_results.rowconfigure(0, weight=1)

# Crear un marco para los gráficos
frame_graphs = ttk.Frame(root, padding="10")
frame_graphs.grid(row=2, column=0, sticky=(tk.W, tk.E))


# Crear el primer gráfico
label = tk.Label(frame_graphs, text="Este gráfico muestra la distribución de las distancias calculadas entre los artículos evaluados y el artículo de referencia.Es útil para entender cómo se distribuyen las similitudes entre los artículos recomendados y el artículo de referencia",wraplength=600, justify="left", font=("Helvetica", 10))
label.grid(row=1, column=0, pady=0, padx=5, sticky='w')
fig_dist = plt.figure(figsize=(6, 4))
canvas_dist = FigureCanvasTkAgg(fig_dist, master=frame_graphs)
canvas_dist.get_tk_widget().grid(row=0, column=0, pady=5, padx=5)

# Crear el segundo gráfico
label = tk.Label(frame_graphs, text="Este gráfico de barras compara las características clave (año de publicación y número de citas) entre el artículo de referencia y los artículos recomendados.Ayuda a visualizar y comparar rápidamente cómo se relacionan los artículos recomendados con respecto al artículo de referencia en términos de sus atributos seleccionados.",wraplength=600, justify="left", font=("Helvetica", 10))
label.grid(row=1, column=1, pady=0, padx=5, sticky='w')
fig_comp = plt.figure(figsize=(6, 4))
canvas_comp = FigureCanvasTkAgg(fig_comp, master=frame_graphs)
canvas_comp.get_tk_widget().grid(row=0, column=1, pady=5, padx=5)



root.mainloop()
