Внедрение правильных тестов для обучающего конвейера.
Задача в том, чтобы идентифицировать такие тесты и заставить набор тестов проходить детерминированно.
Чтобы обучить модель и посмотреть выполнение тестов:
python train.py
python -m pytest example_project/test_basic.py
python test_pipeline.py
python -m pytest --cov=.