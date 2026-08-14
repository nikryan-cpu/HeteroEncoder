"""
Совместимость со старыми pickle-файлами.

pre-trained/vocab.pkl был сериализован до переноса utilities.py в src/ —
pickle хранит класс по имени модуля на момент сохранения ("utilities",
не "src.utilities"), поэтому без этого файла pickle.load падает с
ModuleNotFoundError на уже обученных моделях.

Новый код использует `from src.utilities import SmilesTokenizer` напрямую,
этот файл только для обратной совместимости существующих .pkl.
"""
from src.utilities import SmilesTokenizer  # noqa: F401
