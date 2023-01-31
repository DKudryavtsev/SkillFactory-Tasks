#Импортируем необходимые библиотеки
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import nmslib
import pickle
import plotly.express as px


@st.cache
def read_files(folder_name='data'):
    """
    Функция для чтения файлов.
    Возвращает два DataFrame с рейтингами и характеристиками книг.
    """
    ratings = pd.read_csv(folder_name + '/ratings.csv')
    books = pd.read_csv(folder_name + '/books.csv')
    return ratings, books


def make_mappers(books):
    """
    Функция для создания отображения id в title и authors.
    Возвращает два словаря:
    * Ключи первого словаря — идентификаторы книг, значения — их названия.
    * Ключи второго словаря — идентификаторы книг, значения — их авторы.
    """
    name_mapper = dict(zip(books.book_id, books.title))
    author_mapper = dict(zip(books.book_id, books.authors))

    return name_mapper, author_mapper


def load_embeddings(file_name='item_embeddings.pkl'):
    """
    Функция для загрузки векторных представлений.
    Возвращает прочитанные эмбеддинги книг и индекс (граф) 
    для поиска похожих книг.
    """
    with open(file_name, 'rb') as f:
        item_embeddings = pickle.load(f)

    # Тут мы используем nmslib, чтобы создать быстрый knn
    nms_idx = nmslib.init(method='hnsw', space='cosinesimil')
    nms_idx.addDataPointBatch(item_embeddings)
    nms_idx.createIndex(print_progress=True)
    return item_embeddings, nms_idx


def nearest_books_nms(book_id, index, n=10):
    """
    Функция для поиска ближайших соседей, возвращает построенный индекс.
    Возвращает n наиболее похожих книг и расстояние до них.
    """
    nn = index.knnQuery(item_embeddings[book_id], k=n)
    return nn


def get_recomendation_df(ids, distances, name_mapper, author_mapper):
    """
    Функция для составления таблицы из рекомендованных книг.
    Возвращает DataFrame со столбцами:
    * book_name — название книги;
    * book_author — автор книги;
    * distance — значение метрики расстояния до книги.
    """
    names = []
    authors = []
    #Для каждого индекса книги находим её название и автора
    #Результаты добавляем в списки
    for idx in ids:
        names.append(name_mapper[idx])
        authors.append(author_mapper[idx])
    #Составляем DataFrame
    recomendation_df = pd.DataFrame(
        {'book_name': names, 'book_author': authors, 'distance': distances})
    return recomendation_df


#Загружаем данные
ratings, books = read_files(folder_name='data')
#Создаём словари для сопоставления id книг и их названий/авторов
name_mapper, author_mapper = make_mappers(books)
#Загружаем эмбеддинги и создаём индекс для поиска
item_embeddings, nms_idx = load_embeddings()

st.title("Book Recommendation System")

st.markdown("""Welcome to the web page of a book recommendation app!
This application is a prototype of a recommendation system based on a machine
learning model.

To use the application, you need to 
1. Start by typing an approximate name of the book  
2. Select the exact name in the dropdown menu
3. Specify the number of books you'd like to be recommended

After that, the application will show a list of books most similar to the book 
you have specified""")

#Выбор книги из списка
option = st.selectbox(
    'Select the book',
    ['', *books['title']], 0)  # default value = index 0
#option = st.selectbox("Select the book you need", books['title'].values)

#Проверяем, что поле не пустое
if option != '': 
    #Выводим выбранную книгу
    st.markdown('You have selected: "{}"'.format(option))
    
    #Находим book_id для указанной книги
    val_index = books[books['title'].values == option]['book_id'].values

    #Указываем количество рекомендаций
    count_recomendation = st.number_input(
        label="Specify the number of recommendations you need",
        value=10
    )
    
    #Находим count_recomendation+1 наиболее похожих книг
    ids, distances = nearest_books_nms(
        val_index, nms_idx, count_recomendation+1)
    #Убираем из результатов книгу, по которой производился поиск
    ids, distances = ids[1:], distances[1:]
    
    #Выводим рекомендации к книге
    st.markdown('Most similar books are: ')
    #Составляем DataFrame из рекомендаций
    df = get_recomendation_df(ids, distances, name_mapper, author_mapper)
    #Выводим DataFrame в интерфейсе
    st.dataframe(df[['book_name', 'book_author']])
    
    # Строим столбчатую диаграмму
    fig = px.bar(
        data_frame=df,
        x='book_name',
        y='distance',
        hover_data=['book_author'],
        title='Cosine distance to the nearest books'
    )
    fig.update_xaxes(tickangle=45)
    # Отображаем график в интерфейсе
    st.write(fig)