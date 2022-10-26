# Проект 4. Задача классификации

## Оглавление  
[1. Описание проекта](./README.md#Описание-проекта)  
[2. Какой кейс решаем?](./README.md#Какой-кейс-решаем)  
[3. Краткая информация о данных](./README.md#Краткая-информация-о-данных)  
[4. Этапы работы над проектом](./README.md#Этапы-работы-над-проектом)  
[5. Результаты](./README.md#Результаты)   

### 1. Описание проекта   
Решение задачи машинного обучения, направленной на автоматизацию бизнес-процессов: построение модели, которая будет предсказывать общую продолжительность поездки на такси в Нью-Йорке.

Представьте, что вы заказываете такси из одной точки Нью-Йорка в другую, причём необязательно, что конечная точка должна находиться в пределах города. Сколько вы должны будете заплатить за поездку?

Известно, что стоимость такси в США рассчитывается на основе фиксированной ставки и тарифной стоимости, величина которой зависит от времени и расстояния. Тарифы варьируются в зависимости от города.

В свою очередь, время поездки зависит от множества факторов, таких как направление поездки, время суток, погодные условия и так далее.

Таким образом, если мы разработаем алгоритм, способный определять длительность поездки, мы сможем прогнозировать её стоимость самым тривиальным образом, например, просто умножая стоимость на заданный тариф.

Задача, которую мы будем решать, была представлена в качестве Data Science-соревнования с призовым фондом в 30 000 $ на платформе Kaggle в 2017 году.

:arrow_up: [к оглавлению](./README.md#Оглавление)

### 2. Какой кейс решаем?  

Задача регресии.

**Бизнес-задача:** определить характеристики и с их помощью спрогнозировать длительность поездки на такси.

**Техническая задача специалиста Data Science:** построить модель машинного обучения, которая на основе предложенных характеристик клиента будет предсказывать числовой признак — время поездки такси, то есть решить задачу регрессии.

:arrow_up: [к оглавлению](./README.md#Оглавление)

### 3. Краткая информация о данных
[Исходный датасет](https://drive.google.com/file/d/1X_EJEfERiXki0SKtbnCL9JDv49Go14lF/view?usp=sharing) с данными о поездках.

[Подготовленная выгрузка](https://drive.google.com/file/d/1ecWjor7Tn3HP7LEAm5a0B_wrIfdcVGwR/view?usp=sharing) из [Open Source Routing Machine (OSRM)](https://en.wikipedia.org/wiki/Open_Source_Routing_Machine) - сервиса для построения маршрутов.

[Набор данных](https://lms.skillfactory.ru/assets/courseware/v1/0f6abf84673975634c33b0689851e8cc/asset-v1:SkillFactory+DSPR-2.0+14JULY2021+type@asset+block/weather_data.zip), содержащий информацию о погодных условиях в Нью-Йорке в 2016 году.

[Тестовая выборка](https://drive.google.com/file/d/1C2N2mfONpCVrH95xHJjMcueXvvh_-XYN/view?usp=sharing)

[Подготовленная выгрузка](https://drive.google.com/file/d/1wCoS-yOaKFhd1h7gZ84KL9UwpSvtDoIA/view?usp=sharing) OSRM API для тестовой выборки


:arrow_up: [к оглавлению](./README.md#Оглавление)

### 4. Этапы работы над проектом  

1. Первичная обработка данных
2. Разведывательный анализ данных (EDA)
3. Отбор и преобразование признаков
4. Решение задачи регрессии: линейная регрессия и деревья решений
5. Решение задачи регрессии: ансамбли моделей и построение прогноза

:arrow_up: [к оглавлению](./README.md#Оглавление)

### 5. Результаты  
* Сформирован набор данных на основе нескольких источников информации: исходный датасет; данные маршрута, построенного с помощью сервиса OSRM API; данные о погодных условиях. 
* С помощью Feature Engineering спроектированы новые признаки и выявлены наиболее значимые для построения модели (25 признаков, SelectKBest).
* Исследованы предоставленные данные и продемонстрированы закономерности: зависимость количества и длительности поездок от времени суток и дня недели
* Построены несколько моделей (линейная регрессия, линейная регрессия на полиномиальных признаках, дерево решений, случайный лес, градиентный бустинг), выбрана модель с наилучшим результатом по заданной метрике.
* Спроектирован процесс предсказания длительности поездки для новых данных.
* Решение загружено на платформу Kaggle.

:arrow_up: [к оглавлению](./README.md#Оглавление)