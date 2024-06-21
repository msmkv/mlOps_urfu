[Google disc](https://drive.google.com/drive/folders/1_Oo6kxpLXmGmUeHWIKVzR4QAPgET3ASg?usp=share_link)

Директория [**scripts**](https://github.com/msmkv/mlOps_urfu/tree/main/task4/scripts)

Чтобы запустить:

1. ```git clone```  - клонирование репозитория
2. ```pip install pandas numpy catboost``` - установка зависимостей
3. ```cd scripts``` - переход в директорию с python-скриптами
4. ```python|python3 get_data.py``` - модификация данных #0 - получение данных датасета *titanic*
5. ```python|python3 delete_data.py``` - если данные нужно удалить *(опционально)*
6. ```python|python3 modificator_1.py``` - модификация данных #1 - заполнение колонки **Age** средним значением
7. ```python|python3 modificator_2.py``` - модификация данных #2 - применение *one-hot-encoding* для колонки **Sex**
8. ```dvc push``` - после каждой модификации данных - отправка на google disk через *dvc*

