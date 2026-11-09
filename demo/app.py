"""Демонстрация: вставьте русский текст, и слова, которые стоит пояснить современному читателю, подсветятся.

    pip install -r demo/requirements.txt
    streamlit run demo/app.py

Нужна папка models/ (её создаёт `python -m archword.train_final --raw data/raw --out-dir models`).
"""
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from archword.predict import MODEL_FILE, Scorer, highlight_html, summary_table  # noqa: E402

EXAMPLE = (
    "Ямщик сидел на облучке, а я, закутавшись в шинель, дремал в кибитке. Спросонья послышался мне "
    "голос урядника: «Подорожную!» Я вынул её из кармана и отдал десятнику, который засветил фонарь."
)

st.set_page_config(page_title="archword: какие слова пояснить", page_icon="📖", layout="wide")


@st.cache_resource(show_spinner="Загружаю модель…")
def load_scorer() -> Scorer:
    return Scorer.load(ROOT / "models")


st.title("📖 Какие слова в классике нужно пояснить?")
st.caption(
    "Вставьте отрывок из русской прозы XIX века. Подсвечены слова, которые, по оценке модели, вероятно потребуют "
    "пояснения. Чем ярче подсветка, тем выше вероятность; если навести курсор, видно лемму и оценку."
)

if not (ROOT / "models" / MODEL_FILE).is_file():
    st.error("Не найдена папка models/. Создайте её: `python -m archword.train_final --raw data/raw --out-dir models`.")
    st.stop()

scorer = load_scorer()
meta = scorer.meta
stats = meta.get("threshold_stats", {})

text = st.text_area("Текст", value=EXAMPLE, height=200, max_chars=20000)
left, right = st.columns(2)
with left:
    threshold = st.slider(
        "Порог: подсвечивать слова с вероятностью не ниже", 0.05, 0.9, float(scorer.suggested_threshold), 0.01,
        help="Ниже порог - больше подсвеченных слов (выше полнота, ниже точность).",
    )
with right:
    skip_names = st.checkbox("Не оценивать имена (слова с заглавной буквы не в начале предложения)", value=True)
if stats:
    st.caption(
        f"На размеченных словах при пороге {scorer.suggested_threshold:.2f} модель находила около {stats['recall']:.0%} слов, "
        f"которые читатель счёл незнакомыми, и {stats['precision']:.0%} подсвеченных слов действительно были незнакомы."
    )

if text.strip():
    occurrences = scorer.annotate(text, skip_names=skip_names)
    st.markdown(highlight_html(text, occurrences, threshold), unsafe_allow_html=True)
    table = summary_table(occurrences, threshold)
    st.subheader(f"Слова для пояснения: {len(table)}")
    if table.empty:
        st.write("При таком пороге ничего не подсвечено. Попробуйте снизить порог.")
    else:
        st.dataframe(
            table, hide_index=True, use_container_width=True,
            column_config={"вероятность": st.column_config.ProgressColumn("вероятность", min_value=0.0, max_value=1.0, format="%.2f"),
                           "оценка": st.column_config.NumberColumn("оценка 1-5", format="%.1f")},
        )

with st.expander("Как это работает и чему можно верить"):
    cv = meta.get("cv", {})
    st.markdown(
        f"""
**Модель.** Линейная регрессия (Ridge) по логарифму частоты слова и его леммы (по частотному словарю OpenCorpora и
лемматизатору pymorphy3), длине слова и символьным n-граммам. Обучена на {meta.get('n_train', '?')} словах из «Повестей Белкина»,
размеченных одним читателем по шкале 1-5. Файл модели читается без pickle и считается на numpy.

**Качество.** Кросс-валидация по словам: корреляция Спирмена {cv.get('spearman', '?')}, AUC для порога «оценка не ниже 2»
{cv.get('auc_label_ge2', '?')}. Это умеренное качество: модель заметно лучше простой частоты слова, но не безошибочна.

**Ограничения.**
- Разметка одного человека: модель предсказывает, что незнакомо ему, а не объективную трудность слова.
- Лемма берётся без контекста (самый вероятный разбор), омонимы могут путаться.
- Редкие, но узнаваемые слова (названия, термины) могут получать завышенную оценку, а узнаваемые по виду устаревшие слова - заниженную.
- Обучение на одном произведении; на других авторах качество не проверялось.
"""
    )
