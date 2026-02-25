from transformers import pipeline
from deep_translator import GoogleTranslator
import torch
import re
from gtts import gTTS
import speech_recognition as sr
import os

# --- 1. CONFIGURATION ---
# We use 'LaMini-Flan-T5-248M'. 
# It is the SAME SPEED as the previous one but trained to sound like ChatGPT.
MODEL_NAME = "MBZUAI/LaMini-Flan-T5-248M"

class NeuralRAG:
    def __init__(self):
        print(f"⏳ Loading Local LLM ({MODEL_NAME})...")
        self.llm = pipeline("text2text-generation", model=MODEL_NAME)
        print("✅ Local Brain Loaded!")

    def translate_text(self, text, target_lang='en'):
        try:
            if not text or len(text) < 5: return text
            translator = GoogleTranslator(source='auto', target=target_lang)
            return translator.translate(text)
        except Exception as e:
            return text

    def clean_text(self, text):
        """
        Aggressively removes navigation bars, menus, and short links 
        to ensure the AI only sees real sentences.
        """
        if not text: return ""
        
        # 1. Split into lines to analyze structure
        lines = text.split('\n')
        clean_lines = []
        
        for line in lines:
            # 2. Heuristic: Real content usually has 5+ words or ends in punctuation
            line = line.strip()
            if len(line.split()) > 6 or line.endswith('.'):
                clean_lines.append(line)
        
        # Join back together
        cleaned = " ".join(clean_lines)
        
        # 3. Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()

    def generate_rag_summary(self, query, search_results, target_lang='en'):
        """
        The Core RAG Pipeline:
        1. Aggregates text from search results.
        2. Constructs a specific PROMPT (Question + Context).
        3. Generates a direct answer.
        """
        # 1. Prepare Context
        raw_context = ""
        for result in search_results:
            if result.text:
                # Clean the text to remove menus
                cleaned_snippet = self.clean_text(result.text)
                # We skip the very start (often titles) and take a chunk from the middle/start
                # Taking up to 1000 chars to give the model more substance
                raw_context += cleaned_snippet[:1000] + " "
        
        if len(raw_context) < 50:
            return "No readable content found to analyze. Try a different search."

        # 2. Construct the Prompt (The "Instruction")
        # UPDATED: Instructions now explicitly ask for comprehensive details and steps.
        input_prompt = (
            f"Instruction: Read the context below and provide a comprehensive, detailed answer. "
            f"If the question is about 'how to' build or do something, provide a full step-by-step guide. "
            f"Do not summarize briefly; explain fully.\n\n"
            f"Context: {raw_context[:3000]}\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )
        
        try:
            # 3. Generate Answer
            # UPDATED PARAMETERS:
            # max_length=1024: Effectively "no limit" for this model size. Allows full tutorials.
            # min_length=100: Ensures the answer isn't too short.
            output = self.llm(
                input_prompt, 
                max_length=1024, 
                min_length=100,
                do_sample=True, 
                temperature=0.3,
                repetition_penalty=1.2
            )
            answer_text = output[0]['generated_text']
            
            # 4. Translate if needed
            if target_lang != 'en':
                answer_text = self.translate_text(answer_text, target_lang)
                
            return answer_text
            
        except Exception as e:
            return f"Error generating answer: {e}"

    # --- FEATURE 1: VOICE INTERFACE METHODS ---

    def text_to_speech(self, text, filename="ai_response.mp3"):
        """
        Converts text to an audio file (MP3) using Google TTS.
        """
        try:
            if not text: return None
            # Remove markdown symbols for cleaner speech
            clean_speech = text.replace('*', '').replace('#', '').replace('`', '')
            tts = gTTS(text=clean_speech, lang='en')
            tts.save(filename)
            return filename
        except Exception as e:
            print(f"TTS Error: {e}")
            return None

    def listen_and_transcribe(self):
        """
        Listens to the microphone for 5 seconds and converts speech to text.
        Returns: The recognized text (string) or None.
        """
        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                print("🎤 Adjusting for ambient noise...")
                recognizer.adjust_for_ambient_noise(source, duration=1)
                print("🎤 Listening... (Speak now)")
                
                # Listen for up to 5 seconds of silence, or 10 seconds of speech
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                
                print("🔄 Transcribing...")
                text = recognizer.recognize_google(audio)
                print(f"✅ Heard: {text}")
                return text
        except sr.WaitTimeoutError:
            print("❌ Listening timed out.")
            return None
        except sr.UnknownValueError:
            print("❌ Could not understand audio.")
            return None
        except Exception as e:
            print(f"❌ Microphone Error: {e}")
            return None

if __name__ == "__main__":
    rag = NeuralRAG()
    print("Test Complete.")