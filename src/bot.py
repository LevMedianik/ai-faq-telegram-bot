from __future__ import annotations

import asyncio
import logging

import re
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message

from src.config import settings
from src.dataio import load_faq
from src.embedder import Embedder
from src.index import load_index
from src.retriever import Retriever


log = logging.getLogger("faq-bot")


def _build_faq_map():
    faq_items = load_faq(settings.faq_csv)
    return {x.faq_id: x.answer for x in faq_items}


async def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    if not settings.telegram_token.strip():
        raise ValueError(
            "TELEGRAM_BOT_TOKEN is empty. Put it into .env (see .env.example)."
        )

    # Load FAQ answers
    faq_map = _build_faq_map()
    log.info("Loaded FAQ answers: %d", len(faq_map))

    # Load retrieval index (must be built beforehand)
    embeddings, items_df, meta = load_index(settings.artifacts_dir)
    log.info(
        "Loaded index: items=%d dim=%d model=%s",
        embeddings.shape[0],
        embeddings.shape[1],
        meta.get("model_name"),
    )

    # Create embedder + retriever
    embedder = Embedder(settings.model_name)
    retriever = Retriever(
        embeddings=embeddings,
        items_df=items_df,
        threshold=settings.threshold,
        top_k=settings.top_k,
    )

    bot = Bot(token=settings.telegram_token)
    dp = Dispatcher()

    @dp.message(CommandStart())
    async def on_start(message: Message):
        await message.answer(
            "Привет! Я FAQ-бот поддержки.\n"
            "Задай вопрос — я попробую найти подходящий ответ.\n\n"
            "Команды:\n"
            "/help — справка\n"
            "/ping — проверка\n"
        )

    @dp.message(Command("help"))
    async def on_help(message: Message):
        await message.answer(
            "Я отвечаю на вопросы по базе FAQ.\n"
            "Если уверенность низкая — направляю к оператору.\n\n"
            f"Текущие параметры:\n"
            f"- threshold: {settings.threshold}\n"
            f"- top_k: {settings.top_k}"
        )

    @dp.message(Command("ping"))
    async def on_ping(message: Message):
        await message.answer("pong ✅")
    
    @dp.message(F.text)
    async def on_text(message: Message):
        query = (message.text or "").strip()
        if not query:
            await message.answer("Отправьте текстовый вопрос.")
            return

        # Generic / panic запросы - перевод к оператору
        if is_generic_request(query):
            await message.answer(
                "Похоже на общий запрос без деталей.\n"
                "Перевожу на оператора.\n\n"
                "Пожалуйста, уточните:\n"
                "— что именно не работает;\n"
                "— в каком разделе;\n"
                "— что вы пытались сделать."
            )
            return

        # Embed user query
        q_emb = embedder.encode([query], batch_size=1)[0]

        # Predict
        faq_id, score, confident = retriever.predict_faq(q_emb)

        if confident and faq_id and faq_id in faq_map:
            answer = faq_map[faq_id]
            await message.answer(f"{answer}\n\n(id: {faq_id}, score: {score:.3f})")
        else:
            await message.answer(
                "Не уверен, перевожу на оператора.\n"
                "Оставьте контакты или уточните вопрос.\n\n"
                f"(score: {score:.3f})"
            )
        
    log.info("Bot started.")
    await dp.start_polling(bot)

GENERIC_PATTERN = re.compile(
    r"\b("
    r"помог(ите|и)?|"
    r"срочно|"
    r"проблем(а|ы)|"
    r"не работает|"
    r"ничего не работает|"
    r"сломал(ось|ся)|"
    r"ошибк(а|и)"
    r")\b",
    re.IGNORECASE,
)

# Слова, которые делают запрос "конкретным" (есть объект/операция)
SPECIFIC_HINTS_PATTERN = re.compile(
    r"\b("
    r"парол(ь|я)|"
    r"подписк(а|у|и)|"
    r"тариф(ы|а)?|"
    r"плат(еж|ёж)|оплат|"
    r"чек|счет|счёт|"
    r"email|почт|"
    r"вход|войти|авториз|логин|"
    r"2fa|двухфактор|"
    r"файл|загруз|"
    r"api|ключ|"
    r"команд(а|у)|пользовател"
    r")\b",
    re.IGNORECASE,
)

def is_generic_request(text: str) -> bool:
    """
    Общие/панические запросы -> оператор.
    Ловим "проблема/не работает/ошибка..." и проверяем, есть ли конкретика.
    """
    if not GENERIC_PATTERN.search(text):
        return False

    # Если есть конкретные подсказки (пароль/оплата/2fa/файл/вход...), то это уже не "общий" запрос
    if SPECIFIC_HINTS_PATTERN.search(text):
        return False

    return True

if __name__ == "__main__":
    asyncio.run(main())
