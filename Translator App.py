from googletrans import Translator

def translate_text(text, src='en', dest='es'):
    translator = Translator()
    translation = translator.translate(text, src=src, dest=dest)
    return translation.text

# Example usage
text = 'Hello, how are you?'
translated_text = translate_text(text, dest='es')
print(f'Translated text: {translated_text}')
