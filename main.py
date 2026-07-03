import os
import logging
import requests
import asyncio
from io import BytesIO
from datetime import datetime
from typing import Optional
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    CallbackQueryHandler,
    ConversationHandler
)

# --- Configuration ---
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("No TELEGRAM_BOT_TOKEN set in environment variables")

HF_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")
if not HF_API_KEY:
    logging.warning("No HUGGINGFACE_API_KEY set. Image generation will fail.")

# Hugging Face API endpoints
HF_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"
HF_API_URL_SDXL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"

# Available models
MODELS = {
    "sd21": {
        "name": "Stable Diffusion 2.1",
        "url": "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"
    },
    "sdxl": {
        "name": "Stable Diffusion XL",
        "url": "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    },
    "openjourney": {
        "name": "OpenJourney",
        "url": "https://api-inference.huggingface.co/models/prompthero/openjourney"
    }
}

# --- Logging ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Helper Functions ---
async def generate_image(prompt: str, model_key: str = "sd21") -> BytesIO:
    """
    Sends a prompt to Hugging Face's Stable Diffusion API and returns the image.
    
    Args:
        prompt: The text prompt for image generation
        model_key: The model to use (sd21, sdxl, or openjourney)
    
    Returns:
        BytesIO object containing the generated image
    """
    if not HF_API_KEY:
        raise Exception("Hugging Face API key is missing. Please set HUGGINGFACE_API_KEY.")

    # Get the selected model URL
    model_info = MODELS.get(model_key, MODELS["sd21"])
    api_url = model_info["url"]
    
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}"
    }
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "negative_prompt": "blurry, bad quality, distorted, deformed",
            "num_inference_steps": 30,
            "guidance_scale": 7.5
        }
    }

    try:
        logger.info(f"Generating image with {model_info['name']} for prompt: {prompt[:50]}...")
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            # Success: Return the image as a BytesIO object
            image_bytes = BytesIO(response.content)
            image_bytes.seek(0)
            return image_bytes
        elif response.status_code == 503:
            # Model is loading, wait and retry
            logger.warning("Model is loading, waiting 10 seconds...")
            await asyncio.sleep(10)
            # Retry once
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)
            if response.status_code == 200:
                image_bytes = BytesIO(response.content)
                image_bytes.seek(0)
                return image_bytes
            else:
                raise Exception(f"Model loading failed. Status: {response.status_code}")
        else:
            # Handle API errors
            error_message = f"API Error {response.status_code}: {response.text}"
            logger.error(error_message)
            
            # Check if it's a quota/rate limit issue
            if response.status_code == 429:
                raise Exception("Rate limit exceeded. Please try again in a few minutes.")
            elif response.status_code == 401:
                raise Exception("Invalid API key. Please check your Hugging Face token.")
            else:
                raise Exception(f"Image generation failed: {response.status_code}")
            
    except requests.exceptions.Timeout:
        logger.error("Timeout while calling Hugging Face API")
        raise Exception("Request timed out. Please try a simpler prompt or try again later.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error while calling Hugging Face API: {e}")
        raise Exception("Network error: Could not connect to image generation service.")
    except Exception as e:
        logger.error(f"Unexpected error in generate_image: {e}")
        raise

async def generate_with_progress(update: Update, prompt: str, model_key: str = "sd21") -> None:
    """Process a prompt and generate an image with progress updates."""
    # Send initial status message
    status_msg = await update.message.reply_text(
        f"🎨 Generating image for: '{prompt[:80]}...'\n"
        f"⏳ This may take 30-60 seconds..."
    )
    
    try:
        # Generate the image
        image_bytes = await generate_image(prompt, model_key)
        
        # Update status
        await status_msg.edit_text("✅ Image generated! Sending now...")
        
        # Get model name for caption
        model_name = MODELS.get(model_key, MODELS["sd21"])["name"]
        
        # Send the image with caption
        await update.message.reply_photo(
            photo=image_bytes,
            caption=f"🖼️ Here's your image!\n"
                   f"📝 Prompt: {prompt[:200]}\n"
                   f"🤖 Model: {model_name}\n"
                   f"⏰ {datetime.now().strftime('%H:%M:%S')}"
        )
        
        # Delete status message
        await status_msg.delete()
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error during processing for user {update.effective_user.id}: {error_msg}")
        await status_msg.edit_text(
            f"❌ Sorry, I couldn't generate that image.\n"
            f"Error: {error_msg[:200]}\n\n"
            f"💡 Try using /generate with a simpler prompt."
        )

# --- Bot Command Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for the /start command."""
    user = update.effective_user
    welcome_message = (
        f"🎨 Hello {user.first_name}! I'm AI Painter Bot.\n\n"
        f"I can create stunning AI-generated images from your text descriptions.\n\n"
        f"🛠️ **How to use me:**\n"
        f"• Send `/generate sunset over mountains`\n"
        f"• Or just send me any text message\n"
        f"• Use `/model` to switch between AI models\n"
        f"• Use `/help` for more commands\n\n"
        f"✨ Try it now - just send me a description of what you want to see!"
    )
    await update.message.reply_text(welcome_message)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for the /help command."""
    help_text = (
        f"📖 **Available Commands:**\n\n"
        f"/start - Start the bot\n"
        f"/help - Show this help message\n"
        f"/generate [prompt] - Generate an image from text\n"
        f"/model - Change the AI model\n"
        f"/about - About this bot\n"
        f"/stats - Show your usage stats\n\n"
        f"💡 **Tips:**\n"
        f"• Be specific in your prompts\n"
        f"• Use negative prompts like 'avoid blurry'\n"
        f"• Example: 'a cute cat playing with yarn, detailed, colorful'\n\n"
        f"⚠️ Each generation may take 30-60 seconds"
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")

async def generate_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for the /generate command."""
    if not context.args:
        await update.message.reply_text(
            "❌ Please provide a prompt after the /generate command.\n"
            "Example: `/generate a futuristic cityscape with flying cars`"
        )
        return
    
    prompt = " ".join(context.args)
    # Get selected model from context or use default
    model_key = context.user_data.get("selected_model", "sd21")
    await generate_with_progress(update, prompt, model_key)

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for regular text messages."""
    if update.message.text.startswith('/'):
        return
    prompt = update.message.text
    model_key = context.user_data.get("selected_model", "sd21")
    await generate_with_progress(update, prompt, model_key)

async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for the /model command."""
    keyboard = [
        [InlineKeyboardButton("🖼️ Stable Diffusion 2.1", callback_data="model_sd21")],
        [InlineKeyboardButton("🎨 Stable Diffusion XL", callback_data="model_sdxl")],
        [InlineKeyboardButton("🎭 OpenJourney", callback_data="model_openjourney")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    current_model = context.user_data.get("selected_model", "sd21")
    current_name = MODELS[current_model]["name"]
    
    await update.message.reply_text(
        f"🤖 **Select AI Model**\n\n"
        f"Current: **{current_name}**\n\n"
        f"Choose a model for image generation:",
        reply_markup=reply_markup,
        parse_mode="Markdown"
    )

async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for the /about command."""
    about_text = (
        f"🤖 **AI Painter Bot**\n"
        f"Version: 1.0.0\n\n"
        f"Powered by Hugging Face's Stable Diffusion models.\n\n"
        f"👨‍💻 Built with Python and python-telegram-bot library.\n"
        f"🚀 Deployed on Railway\n\n"
        f"📊 Free tier limitations:\n"
        f"• Rate limited to ~5 images per minute\n"
        f"• Generations may take 30-60 seconds\n\n"
        f"⚠️ **Note**: I'm running on free APIs, so responses might be slower."
    )
    await update.message.reply_text(about_text, parse_mode="Markdown")

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for the /stats command."""
    user_id = update.effective_user.id
    # Initialize stats if not exists
    if "stats" not in context.user_data:
        context.user_data["stats"] = {"generations": 0, "last_used": None}
    
    stats = context.user_data["stats"]
    await update.message.reply_text(
        f"📊 **Your Stats**\n\n"
        f"Total generations: {stats['generations']}\n"
        f"Last use: {stats['last_used'] or 'Never'}\n\n"
        f"Keep creating! 🎨"
    )

async def handle_model_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for model selection callback."""
    query = update.callback_query
    await query.answer()
    
    # Extract model key from callback data
    model_key = query.data.split("_")[1]
    context.user_data["selected_model"] = model_key
    
    model_name = MODELS[model_key]["name"]
    await query.edit_message_text(
        f"✅ Model switched to: **{model_name}**\n\n"
        f"Try generating an image now! 🎨",
        parse_mode="Markdown"
    )

# --- Main Function ---
def main() -> None:
    """Start the bot and set up handlers."""
    # Create the Application
    application = Application.builder().token(BOT_TOKEN).build()

    # Register command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("generate", generate_command))
    application.add_handler(CommandHandler("model", model_command))
    application.add_handler(CommandHandler("about", about_command))
    application.add_handler(CommandHandler("stats", stats_command))

    # Register handlers for text messages and callbacks
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    application.add_handler(CallbackQueryHandler(handle_model_callback, pattern="^model_"))

    # Start the bot
    logger.info("🤖 AI Painter Bot is starting...")
    
    # For Railway, we use polling
    application.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True
    )

if __name__ == "__main__":
    main()
