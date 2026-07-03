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

# --- Image Generation Function ---
async def generate_image(prompt: str, model_key: str = "sd21") -> BytesIO:
    """
    Generate image using Hugging Face API.
    
    Args:
        prompt: Text description of the image
        model_key: Model to use (sd21, sdxl, or openjourney)
    
    Returns:
        BytesIO object containing the generated image
    
    Raises:
        Exception: If image generation fails
    """
    if not HF_API_KEY:
        raise Exception("Hugging Face API key is missing. Please set HUGGINGFACE_API_KEY.")

    model_info = MODELS.get(model_key, MODELS["sd21"])
    api_url = model_info["url"]
    
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}"
    }
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "negative_prompt": "blurry, bad quality, distorted, deformed, ugly",
            "num_inference_steps": 30,
            "guidance_scale": 7.5
        }
    }

    try:
        logger.info(f"Generating image with {model_info['name']} for prompt: {prompt[:50]}...")
        
        # First attempt
        response = requests.post(api_url, headers=headers, json=payload, timeout=90)
        
        # Handle model loading (503)
        if response.status_code == 503:
            logger.warning("Model is loading, waiting 15 seconds...")
            await asyncio.sleep(15)
            # Retry after waiting
            response = requests.post(api_url, headers=headers, json=payload, timeout=90)
        
        # Check response
        if response.status_code == 200:
            image_bytes = BytesIO(response.content)
            image_bytes.seek(0)
            return image_bytes
        else:
            # Handle specific error codes
            if response.status_code == 429:
                raise Exception("Rate limit exceeded. Please wait a few minutes and try again.")
            elif response.status_code == 401:
                raise Exception("Invalid Hugging Face API key. Please check your token.")
            elif response.status_code == 503:
                raise Exception("Model is still loading. Please try again in 30 seconds.")
            else:
                error_msg = f"API Error {response.status_code}: {response.text[:200]}"
                logger.error(error_msg)
                raise Exception(f"Image generation failed: {response.status_code}")
            
    except requests.exceptions.Timeout:
        raise Exception("Request timed out after 90 seconds. Please try a simpler prompt.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error: {e}")
        raise Exception(f"Network error: Could not connect to image generation service.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

async def generate_with_progress(update: Update, prompt: str, model_key: str = "sd21") -> None:
    """
    Generate image with progress updates.
    
    Args:
        update: Telegram update object
        prompt: Image description
        model_key: Model to use
    """
    # Send initial status
    status_msg = await update.message.reply_text(
        f"🎨 **Generating image...**\n\n"
        f"📝 Prompt: `{prompt[:100]}{'...' if len(prompt) > 100 else ''}`\n"
        f"🤖 Model: {MODELS.get(model_key, MODELS['sd21'])['name']}\n"
        f"⏳ This may take 30-60 seconds...",
        parse_mode="Markdown"
    )
    
    try:
        # Generate the image
        image_bytes = await generate_image(prompt, model_key)
        
        # Update status
        await status_msg.edit_text(
            f"✅ **Image generated successfully!**\n\n"
            f"📝 Prompt: `{prompt[:150]}{'...' if len(prompt) > 150 else ''}`",
            parse_mode="Markdown"
        )
        
        # Get model name for caption
        model_name = MODELS.get(model_key, MODELS["sd21"])["name"]
        
        # Send the image
        await update.message.reply_photo(
            photo=image_bytes,
            caption=f"🖼️ **Here's your image!**\n\n"
                   f"📝 Prompt: {prompt[:200]}{'...' if len(prompt) > 200 else ''}\n"
                   f"🤖 Model: {model_name}\n"
                   f"⏰ {datetime.now().strftime('%H:%M:%S')}\n\n"
                   f"💡 Try a different prompt or use /model to switch AI models!",
            parse_mode="Markdown"
        )
        
        # Delete status message
        await status_msg.delete()
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error during generation: {error_msg}")
        await status_msg.edit_text(
            f"❌ **Sorry, I couldn't generate that image.**\n\n"
            f"Error: `{error_msg[:200]}`\n\n"
            f"💡 **Tips:**\n"
            f"• Try a simpler prompt\n"
            f"• Wait a moment and try again\n"
            f"• Use /help for more commands",
            parse_mode="Markdown"
        )

# --- Bot Command Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for /start command."""
    user = update.effective_user
    welcome_message = (
        f"🎨 **Hello {user.first_name}! I'm AI Painter Bot.**\n\n"
        f"I create stunning AI-generated images from your text descriptions.\n\n"
        f"🛠️ **How to use me:**\n"
        f"• Send `/generate sunset over mountains`\n"
        f"• Or just send me any text message\n"
        f"• Use `/model` to switch AI models\n"
        f"• Use `/help` for more commands\n\n"
        f"✨ **Try it now!** Just send me a description of what you want to see."
    )
    await update.message.reply_text(welcome_message, parse_mode="Markdown")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for /help command."""
    help_text = (
        f"📖 **Available Commands:**\n\n"
        f"`/start` - Start the bot\n"
        f"`/help` - Show this help message\n"
        f"`/generate [prompt]` - Generate an image from text\n"
        f"`/model` - Change the AI model\n\n"
        f"💡 **Tips for Better Results:**\n"
        f"• Be specific and descriptive\n"
        f"• Mention style, colors, and details\n"
        f"• Example: `a cute orange cat playing with yarn, detailed, colorful`\n"
        f"• Mention negative things to avoid\n\n"
        f"⚠️ **Limitations:**\n"
        f"• Each generation takes 30-60 seconds\n"
        f"• Free API has rate limits\n"
        f"• Max prompt length: 2000 characters"
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")

async def generate_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for /generate command."""
    if not context.args:
        await update.message.reply_text(
            f"❌ **Please provide a prompt.**\n\n"
            f"Example: `/generate a futuristic cityscape with flying cars`\n"
            f"Or: `/generate a beautiful sunset over a mountain lake`",
            parse_mode="Markdown"
        )
        return
    
    prompt = " ".join(context.args)
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
    """Handler for /model command."""
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

async def handle_model_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for model selection callback."""
    query = update.callback_query
    await query.answer()
    
    model_key = query.data.split("_")[1]
    context.user_data["selected_model"] = model_key
    
    model_name = MODELS[model_key]["name"]
    await query.edit_message_text(
        f"✅ **Model switched to: {model_name}**\n\n"
        f"Try generating an image now! Just send me a description. 🎨",
        parse_mode="Markdown"
    )

# --- Main Function ---
def main() -> None:
    """Start the bot."""
    if not BOT_TOKEN:
        logger.error("No TELEGRAM_BOT_TOKEN found!")
        return
    
    logger.info("🤖 AI Painter Bot is starting...")
    logger.info(f"Available models: {', '.join(MODELS.keys())}")
    
    # Create the Application
    application = Application.builder().token(BOT_TOKEN).build()

    # Register command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("generate", generate_command))
    application.add_handler(CommandHandler("model", model_command))

    # Register handlers for text messages and callbacks
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    application.add_handler(CallbackQueryHandler(handle_model_callback, pattern="^model_"))

    # Start the bot
    try:
        application.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True
        )
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")
        raise

if __name__ == "__main__":
    main()
