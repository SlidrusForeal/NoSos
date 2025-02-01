import asyncio
import asyncpg
import os
import logging
import yaml
from datetime import datetime
from telegram import Update, Bot
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# Читаем переменные окружения
DATABASE_URL = os.getenv("DATABASE_URL")  # PostgreSQL URL
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID")

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DatabaseManager:
    def __init__(self, db_url):
        self.db_url = db_url
        self.pool = None

    async def connect(self):
        self.pool = await asyncpg.create_pool(self.db_url)
        await self.create_tables()

    async def create_tables(self):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id BIGINT PRIMARY KEY,
                    username TEXT,
                    approved BOOLEAN DEFAULT FALSE,
                    subscribed BOOLEAN DEFAULT TRUE
                );
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id SERIAL PRIMARY KEY,
                    message TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT NOW()
                );
            """)

    async def add_user(self, user_id: int, username: str):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO users (user_id, username, approved, subscribed)
                VALUES ($1, $2, FALSE, TRUE)
                ON CONFLICT (user_id) DO NOTHING;
            """, user_id, username)

    async def approve_user(self, user_id: int):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE users SET approved = TRUE WHERE user_id = $1;
            """, user_id)

    async def get_users(self):
        async with self.pool.acquire() as conn:
            return await conn.fetch("SELECT * FROM users;")

    async def save_alert(self, message: str):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO alerts (message) VALUES ($1);
            """, message)

class TelegramBot:
    def __init__(self, token, db_manager):
        self.bot = Bot(token=token)
        self.db = db_manager
        self.app = ApplicationBuilder().token(token).build()
        self._register_handlers()

    def _register_handlers(self):
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("approve", self.approve))
        self.app.add_handler(CommandHandler("users", self.list_users))

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        await self.db.add_user(user.id, user.username)
        await update.message.reply_text("Ваш запрос на одобрение отправлен админу!")
        await self.bot.send_message(chat_id=ADMIN_CHAT_ID, text=f"Новый пользователь: {user.username} ({user.id})")

    async def approve(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args:
            await update.message.reply_text("Используйте: /approve <user_id>")
            return
        user_id = int(context.args[0])
        await self.db.approve_user(user_id)
        await update.message.reply_text(f"Пользователь {user_id} одобрен")
        await self.bot.send_message(chat_id=user_id, text="Ваш аккаунт одобрен!")

    async def list_users(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        users = await self.db.get_users()
        message = "Список пользователей:\n"
        for user in users:
            message += f"{user['user_id']} - {user['username']} - {'Одобрен' if user['approved'] else 'Ожидает'}\n"
        await update.message.reply_text(message)

    def run(self):
        self.app.run_polling()

async def main():
    db = DatabaseManager(DATABASE_URL)
    await db.connect()
    bot = TelegramBot(TELEGRAM_TOKEN, db)
    bot.run()

if __name__ == "__main__":
    asyncio.run(main())
