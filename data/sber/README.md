# Корпус: документация СберБизнеса

Сайт sberbank.ru запрещает автоматический сбор (robots.txt), поэтому краулера в проекте нет.
Страницы сохраняются вручную из браузера — так же, как это сделал бы аналитик, собирая корпус для PoC.
**Сохранённые страницы в git не коммитятся** (см. `.gitignore`): в репозитории только код, каталог
сущностей и вопросы golden dataset.

## Что сохранить (≈15–25 страниц)

Откройте страницу, раскройте все аккордеоны и вкладки с условиями, затем
`Ctrl+S → «Веб-страница, только HTML»` или `Ctrl+P → «Сохранить как PDF»`.
Тарифные PDF-файлы с тех же страниц скачиваются как есть.

| Раздел | Страница |
|---|---|
| РКО | https://www.sberbank.ru/ru/s_m_business/bankingservice/rko |
| Тарифы РКО | https://www.sberbank.ru/ru/s_m_business/bankingservice/rko/tariffs |
| Пакеты услуг | https://www.sberbank.ru/ru/s_m_business/bankingservice/rko/tariffs/pu |
| Эквайринг | https://www.sberbank.ru/ru/s_m_business/bankingservice/acquiring_total |
| Кредиты | https://www.sberbank.ru/ru/s_m_business/credits |
| Зарплатный проект | https://www.sberbank.ru/ru/s_m_business/bankingservice/cards/salaryproject |
| Статьи «Про бизнес» | https://www.sberbank.ru/ru/s_m_business/pro_business/chto-takoe-rko и соседние статьи раздела |

Добавьте 2–3 страницы конкретных продуктов внутри каждого раздела (отдельный тариф, отдельный вид эквайринга, отдельный кредит) —
на них и проверяется сопоставление сущностей.

Имена файлов станут `doc_id` в golden dataset: называйте коротко и латиницей — `rko_tariffs.html`,
`acquiring_qr.html`, `credit_oborot.pdf`.

## Порядок работы

```bash
# 1. положить файлы в data/sber/docs/
# 2. посмотреть, как документы разбились на разделы и чанки
python -m smbrag ask -c configs/sber.yaml "Сколько стоит обслуживание счёта?"

# 3. черновик каталога сущностей: названия в кавычках после «тариф/пакет/кредит…»
python -m smbrag suggest-entities -c configs/sber.yaml
#    → перенести в data/sber/entities.yaml, добавить алиасы (как пишут клиенты)

# 4. написать 40–60 вопросов в eval/golden_sber.jsonl (шаблон — eval/golden_sber.template.jsonl)
#    и проверить разметку против корпуса
python -m smbrag validate -c configs/sber.yaml --golden eval/golden_sber.jsonl

# 5. прогон (в Colab — с Qwen, см. notebooks/)
python -m smbrag eval -c configs/sber.yaml --golden eval/golden_sber.jsonl --out reports/sber
```

## Как составлять golden dataset

* **single** — один факт из одного раздела («сколько стоит обслуживание на пакете X»).
* **paraphrase** — вопрос словами клиента, без терминов из документа («сколько стоит приём оплаты на сайте» вместо «интернет-эквайринг»).
* **typo** — опечатки в названии продукта.
* **entity_confusion** — вопрос про продукт, у которого есть «соседи» с похожими условиями; главный источник галлюцинаций.
* **multi_hop** — ответ требует двух фактов или условия («сколько будет стоить при 3 сотрудниках»).
* **unanswerable** — правдоподобный вопрос, ответа на который в сохранённых страницах нет. Их должно быть 15–20%:
  без них не измерить, выдумывает ли ассистент.

`evidence` — короткая подстрока из документа (5–40 символов), по которой видно, что чанк релевантен.
`expected_facts` — то, без чего ответ неверен; варианты записи через `|` (`1 490|1490`).
Факты сверяйте с сохранённой страницей и указывайте дату сохранения: условия банков меняются.
