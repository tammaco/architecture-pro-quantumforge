import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from rag_bot import RAGBot

load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
BASE_URL = "https://telegram-api-proxy-anonymous.pages.dev/api/bot"

rag_bot = RAGBot()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Бот работает")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    response = rag_bot.ask(user_message)
    answer = response["answer"]
    sources = response.get("sources", [])
    
    if sources:
        answer += f"\n\nИсточники: {', '.join(sources)}"
    
    await update.message.reply_text(answer)

def main():
    app = Application.builder().token(TOKEN).base_url(BASE_URL).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    print("Бот запущен")
    app.run_polling()

if __name__ == "__main__":
    main()