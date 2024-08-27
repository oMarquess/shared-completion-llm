import logging
import os
from dotenv import load_dotenv
import openai
from pydantic import BaseModel
from fastapi import HTTPException
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Load environment variables
load_dotenv(override=True)

# Environment variables for API keys and model configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# Initialize logging
logging.basicConfig(level=logging.INFO)

class ChatOpenAILLM:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._initialize_openai()
        
        # Load templates during initialization
        self.system_message_template = self._load_template("../prompt_templates/system_prompt.txt")
        self.user_message_template = self._load_template("../prompt_templates/user_prompt.txt")

    def _initialize_openai(self):
        try:
            # Initialize OpenAI API key
            openai.api_key = OPENAI_API_KEY
            self.logger.info("OpenAI initialized successfully! 🎉")  # Added emoji
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to initialize OpenAI")

    def create_joke(self, word: str) -> str:
        try:
            # Update user message template with the word
            user_message_template = self.user_message_template.format(word=word)
            prompt = self._generate_prompt(user_message_template)
            self.logger.info(f"Generated prompt: {prompt} 😊")  # Added emoji

            response = openai.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "system", "content": prompt}],
                max_tokens=100,
                temperature=0.7,
            )
            # Access the content of the response using dot notation
            joke = response.choices[0].message.content.strip()
            self.logger.info(f"Generated joke: {joke} 😂")  # Added emoji

            return joke
        except Exception as e:
            self.logger.error(f"Error generating joke: {str(e)} ❌")  # Added emoji
            raise HTTPException(status_code=500, detail="Failed to generate joke")

    def _generate_prompt(self, user_message_template: str) -> str:
        chat_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("{system_message}"),
            HumanMessagePromptTemplate.from_template("{user_input}"),
        ])

        messages = chat_template.format_messages(
            system_message=self.system_message_template,
            user_input=user_message_template
        )

        return "\n".join([message.content for message in messages])

    def _load_template(self, filepath: str) -> str:
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                return file.read().strip()
        except Exception as e:
            self.logger.error(f"Failed to load template from {filepath}: {str(e)} ⚠️")  # Added emoji
            raise HTTPException(status_code=500, detail="Failed to load prompt template")