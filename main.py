import os
import logging
import requests
import asyncio
from io import BytesIO
from datetime import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    CallbackQueryHandler
)

# --- Configuration ---
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("No TELEGRAM_BOT_TOKEN set in environment variables")

HF_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")
if not HF_API_KEY:
    logging.warning("No HUGGINGFACE_API_KEY set. Image generation will fail.")

# Hugging Face API endpoints
MODELS = {
    "sd21": {
        "name": "Stable Diffusion 2.1",
        "url": "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"
    },
    "sdxl": {
        "name": "Stable Diffusion XL",
        "url": "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    }
}

# --- Logging ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Image Generation ---
async def generate_image(prompt: str, model_key: str = "sd21") -> BytesIO:
    """Generate image using Hugging Face API."""
    if not HF_API_KEY:
        raise Exception("Hugging Face API key is missing")

    model_info = MODELS.get(model_key, MODELS["sd21"])
    api_url = model_info["url"]
    
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "negative_prompt": "blurry, bad quality, distorted, deformed",
            "num_inference_steps": 30,
            "guidance_scale": 7.5
        }
    }

    try:
        logger.info(f"Generating image with {model_info['name']}")
        response = requests.post(api_url, headers=headers, json=payload, timeout=90)
        
        if response.status_code == 503:
            logger.warning("Model loading, waiting...")
            await asyncio.sleep(15)
            response = requests.post(api_url, headers=headers, json=payload, timeout=90)
        
        if response.status_code == 200:
            image_bytes = BytesIO(response.content)
            image_bytes.seek(0)
            return image_bytes
        else:
            error_msg = f"API Error: {response.status_code}"
            logger.error(error_msg)
            raise Exception(error_msg)
            
    except requests.exceptions.Timeout:
        raise Exception("Request timed out. Try a simpler prompt.")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error: {str(e)}")
    except Exception as e:
        raise Exception(f"Generation failed: {str(e)}")

async def generate_with_progress(update: Update, prompt: str, model_key: str = "sd21") -> None:
    """Generate image with progress updates."""
    status_msg = await update.message.reply_text(
        f"🎨 Generating image...\n⏳ This may take 30-60 seconds"
    )
    
    try:
        image_bytes = await generate_image(prompt, model_key)
        model_name = MODELS.get(model_key, MODELS["sd21"])["name"]
        
        await update.message.reply_photo(
            photo=image_bytes,
            caption=f"🖼️ Here's your image!\n📝 {prompt[:150]}\n🤖 {model_name}"
        )
        await status_msg.delete()
        
    except Exception as e:
        await status_msg.edit_text(f"❌ Error: {str(e)[:200]}")

# --- Bot Commands ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for /start command."""
    user = update.effective_user
    await update.message.reply_text(
        f"🎨 Hello {user.first_name}! I'm AI Painter Bot.\n\n"
        f"Send me a description of what you want to see!\n"
        f"Example: 'a beautiful sunset over mountains'\n\n"
        f"Commands:\n"
        f"/generate [prompt] - Generate image\n"
        f"/model - Change AI model\n"
        f"/help - Show help"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for /help command."""
    await update.message.reply_text(
        f"📖 How to use:\n\n"
        f"1. Send any text description\n"
        f"2. Or use /generate [prompt]\n"
        f"3. Use /model to switch models\n\n"
        f"💡 Tip: Be specific for better results!\n"
        f"Example: 'a cute orange cat playing with yarn'"
    )

async def generate_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for /generate command."""
    if not context.args:
        await update.message.reply_text(
            "❌ Please provide a prompt.\n"
            "Example: /generate a futuristic cityscape"
        )
        return
    
    prompt = " ".join(context.args)
    model_key = context.user_data.get("selected_model", "sd21")
    await generate_with_progress(update, prompt, model_key)

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for text messages."""
    if update.message.text.startswith('/'):
        return
    
    prompt = update.message.text
    model_key = context.user_data.get("selected_model", "sd21")
    await generate_with_progress(update, prompt, model_key)

async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for /model command."""
    keyboard = [
        [InlineKeyboardButton("🖼️ SD 2.1", callback_data="model_sd21")],
        [InlineKeyboardButton("🎨 SD XL", callback_data="model_sdxl")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "🤖 Select AI Model:",
        reply_markup=reply_markup
    )

async def handle_model_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for model selection."""
    query = update.callback_query
    await query.answer()
    
    model_key = query.data.split("_")[1]
    context.user_data["selected_model"] = model_key
    model_name = MODELS[model_key]["name"]
    
    await query.edit_message_text(f"✅ Model switched to: {model_name}")

# --- Main ---
def main() -> None:
    """Start the bot."""
    if not BOT_TOKEN:
        logger.error("No BOT_TOKEN found!")
        return
    
    logger.info("🤖 AI Painter Bot starting...")
    
    application = Application.builder().token(BOT_TOKEN).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("generate", generate_command))
    application.add_handler(CommandHandler("model", model_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    application.add_handler(CallbackQueryHandler(handle_model_callback, pattern="^model_"))
    
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
