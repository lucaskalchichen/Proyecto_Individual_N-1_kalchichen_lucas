# Funciones del Proyecto Final Individual Nº 1 - MLOps

#Importacion de Librerías
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import pandas as pd 
import numpy as np 
import uvicorn
import operator
import ast


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel

# Instanciamos la aplicación
app = FastAPI()  #http://127.0.0.1:8000

#Cargamos los dataframes de la funciones limpios

df_f1_dev = pd.read_parquet(r'dataset/Funcciones_dataset/fun_1_dev.parquet')
df_f2_user = pd.read_parquet(r'dataset/Funcciones_dataset/fun_2_user_games.parquet')
df_f2_user_games = pd.read_parquet(r'dataset/Funcciones_dataset/fun_2_user_games.parquet')
df_f2_user_reviews = pd.read_parquet(r'dataset/Funcciones_dataset/fun_2_user_rew.parquet')
df_f2_games_price = pd.read_parquet(r'dataset/Funcciones_dataset/fun_2_games_price.parquet')
df_f3_genres = pd.read_parquet(r'dataset/Funcciones_dataset/fun_3_genres.parquet')
df_f3_users = pd.read_parquet(r'dataset/Funcciones_dataset/fun_3_users.parquet')
df_f4_rv = pd.read_parquet(r'dataset/Funcciones_dataset/fun_4_rv.parquet')
df_f4_reviewed_games = pd.read_parquet(r'dataset/Funcciones_dataset/fun_4_reviewed_g.parquet')
df_f5_devs = pd.read_parquet(r'dataset/Funcciones_dataset/fun_5_devs.parquet')
df_f5_reviewed_games = pd.read_parquet(r'dataset/Funcciones_dataset/fun_5_reviewd_games.parquet')
df_f6_recomendaciones =pd.read_parquet(r'dataset/Funcciones_dataset/fun_6_recomendations.parquet')

### Funciones para Alimentar la API

@app.get("/", response_class=HTMLResponse)
async def inicio():
    template = """
    <!DOCTYPE html>
    <html>
        <head>
            <title>API Steam</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    padding: 20px;
                }
                h1 {
                    color: #333;
                    text-align: center;
                }
                p {
                    color: #666;
                    text-align: center;
                    font-size: 18px;
                    margin-top: 20px;
                }
            </style>
        </head>
        <body>
            <h1>API de consultas de la plataforma STEAM</h1>
            <p>Bienvenido a la API de STEAMGAMES, su fuente confiable para consultas especializadas sobre la plataforma de videojuegos.</p>
        </body>
    </html>
    """
    return HTMLResponse(content=template)


@app.get('/developer/{dev}')
def developer(dev: str):
  df_dev = df_f1_dev.copy()
  df_dev = df_dev[df_dev['developer'] == dev]
    
    # Calculate the number of items per year
  items_per_year = df_dev.groupby('year').size()
    
    # Calculate the percentage of free content per year
  free_content_per_year = df_dev[df_dev['price'] == 0.0].groupby('year').size() / items_per_year.replace(0, np.nan) * 100
  free_content_per_year = free_content_per_year.fillna(0)  # Replace NaN values with 0
    
  data = []
  for year in items_per_year.index:
      data.append([year, items_per_year[year], free_content_per_year.get(year, 0)])  # Retrieve the value or default to 0
        
  data_return = pd.DataFrame(data, columns=['year', 'items', 'free_content'])
  return data_return

@app.get("/userdata/{user_id}")
def userdata(user_id: str):

  df_games_price = df_f2_games_price.copy()
  df_user_games = df_f2_user_games.copy()
  df_user_reviews = df_f2_user_reviews.copy()

  # Filtrar los juegos que posee el usuario en el DataFrame df_games_price
  df_user_games_price = df_games_price[df_games_price['item_id'].isin(df_user_games[df_user_games['user_id'] == user_id]['item_id'])]
    
  # Filtrar las reviews hechas por el usuario en el DataFrame df_reviews
  df_user_reviews = df_user_reviews[df_user_reviews['user_id'] == user_id]
    
  # Calcular el dinero total gastado por el usuario
  total_money_spent = df_user_games_price['price'].sum()
    
  # Calcular el porcentaje de recomendación basado en reviews.recommend
  percentage_recommendation = df_user_reviews['recommend'].mean() * 100 if not df_user_reviews.empty else 0
    
  # Obtener el número de items
  num_items = len(df_user_games_price)
  #return df_user_reviews['recommend'].mean()
  return {"Usuario": user_id, "Dinero gastado": total_money_spent, "% de recomendación": f"{percentage_recommendation}%", "Cantidad de items": num_items}

@app.get('/UserForGenre/{genero}')
def UserForGenre(genero: str):

  df_genres = df_f3_genres.copy()
  df_users = df_f3_users.copy()
  # Filtrar el DataFrame para dejar solo los juegos que contengan el género especificado
  df_genre = df_genres[df_genres['genres'].apply(lambda x: genero in x if isinstance(x, list) else False)]

  # Filtrar los usuarios que poseen los juegos del género específico
  df_user_aggregated = df_users[df_users['item_id'].isin(df_genre['item_id'])]
    
  # Merge para concatenar el año de df_genre a df_user_aggregated basado en el item_id
  df_user_aggregated = df_user_aggregated.merge(df_genre[['item_id', 'year']], on='item_id', how='left')
    
  # Calcular la suma de las horas jugadas por cada usuario a los juegos del género específico
  user_hours_per_game = df_user_aggregated.groupby('user_id')['hours_game'].sum()
    
  # Obtener al usuario con más horas jugadas
  user_most_hours_user_id = user_hours_per_game.idxmax()
    
  # Filtrar las horas jugadas por el usuario con más horas jugadas
  user_most_hours_df = df_user_aggregated[df_user_aggregated['user_id'] == user_most_hours_user_id]
    
  # Calcular la cantidad de horas jugadas por año del usuario con más horas jugadas considerando el año de publicación del juego
  hours_per_year = user_most_hours_df.groupby('year')['hours_game'].sum().reset_index()
    
  # Formatear el resultado en el formato especificado
  result = {
        "Usuario con más horas jugadas para " + genero: user_most_hours_user_id,
        "Horas jugadas": [{"Año": int(row['year']), "Horas": int(row['hours_game'])} for index, row in hours_per_year.iterrows()]
  }
    
  return result


@app.get('/best_developer_year/{year}')
def best_developer_year(year: int):
  df_reviewed_games = df_f4_reviewed_games
  df_rv = df_f4_rv.copy()
  df_reviewed_games = df_reviewed_games[df_reviewed_games['year'] == year]  # Filtrar por el año dado
  df_reviewed_games = df_reviewed_games.merge(df_rv, on='item_id')  # Combinar con el DataFrame de reviews
    
  # Filtrar por reviews positivas (recommend = True y sentiment_analysis = 2)
  positive_reviews = df_reviewed_games[(df_reviewed_games['recommend'] == True) & (df_reviewed_games['sentiment_analysis'] == 2)]
    
  # Contar los juegos más recomendados por usuarios por desarrollador
  top_developers = positive_reviews.groupby('developer').size().nlargest(3)
    
    # Crear el formato de retorno
  resultado = [{"Puesto 1" : top_developers.index[0]}, {"Puesto 2" : top_developers.index[1]}, {"Puesto 3" : top_developers.index[2]}]
    
  return resultado

@app.get('/developer_reviews_analysis/{desarrolladora}')
def developer_reviews_analysis(desarrolladora: str):
  df_dev = df_f5_devs.copy()
  df_reviews = df_f5_reviewed_games.copy()
  df_dev = df_dev[df_dev['developer'] == desarrolladora]
  df_dev_reviews = df_reviews[df_reviews['item_id'].isin(df_dev['item_id'])]
    
  # Filtrar las reviews con análisis de sentimiento positivo y negativo
  positive_reviews = df_dev_reviews[df_dev_reviews['sentiment_analysis'] == 2]
  negative_reviews = df_dev_reviews[df_dev_reviews['sentiment_analysis'] == 1]
    
  # Contar el total de reviews positivas y negativas por desarrollador
  positive_count = len(positive_reviews)
  negative_count = len(negative_reviews)
    
  return {desarrolladora: ['Negative = ' + str(negative_count), 'Positive = ' + str(positive_count)]}

@app.get('/recomendacion/{juego}')
def recomendacion_juego(juego):
  
  df_recomendacion = df_f6_recomendaciones.copy()
    
  vector_gradiente = TfidfVectorizer(min_df=10, max_df=0.5, ngram_range=(1,2), stop_words='english')
    
  df_recomendations_transform = vector_gradiente.fit_transform(df_recomendacion['genres'])

  cosineSim = linear_kernel(df_recomendations_transform[:1000,:])

  indices = pd.Series(df_recomendacion.index, index=df_recomendacion['item_id']).drop_duplicates()

  idx = indices[juego]

  recomendacones = list(enumerate(cosineSim[idx]))

  recomendacones = sorted(recomendacones, key=lambda x: x[1], reverse=True)

  recomendacones = recomendacones[1:6]

  itemIndices = [i[0] for i in recomendacones]

  resultado=df_recomendacion['name'].iloc[itemIndices].values
    
  return dict(enumerate(resultado.flatten(), 1))


