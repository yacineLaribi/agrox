import os
import json
import google.generativeai as genai
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# --- Configuration ---
# NOTE: The actual API key is expected to be loaded by the environment.

@csrf_exempt
def chat_with_bot(request):
    """
    Handles user messages, communicates with the Gemini API, and returns the bot's response.
    """
    if request.method == 'POST':
        try:
            # 1. Load data from the request body
            data = json.loads(request.body)
            # The backend only needs the 'message' field, as verified by the client-side fix.
            user_message = data.get('message')

            if not user_message:
                return JsonResponse({'error': 'Message cannot be empty.'}, status=400)

            # 2. Configure the API key
            api_key = os.environ.get('GOOGLE_API_KEY')
            if not api_key:
                # IMPORTANT: Ensure your server environment variable is correctly set
                return JsonResponse({'error': 'GOOGLE_API_KEY environment variable not set.'}, status=500)
            
            genai.configure(api_key=api_key)

            # 3. FIX: Use the standard and robust model name "gemini-2.5-flash"
            model = genai.GenerativeModel("gemini-2.5-flash")

            system_instruction = (
                "You are an agricultural assistant specialized in plant hybridization and breeding, named AGROX. "
                "You are an expert in botany, plant genetics, reproduction, and both interspecific and intergeneric hybridization. "
                "Your explanations are always clear, precise, and detailed. "
                "Respond to the user's message in plain text only, using Markdown when necessary. "
                "If you do not understand the request or if the question is not related to agriculture, botany, "
                "plant biology, genetics, or hybridization, state that you do not know. "
                "Do not use any greetings. Do not use HTML tags, images, or links."
            )

            # 4. Pass the system instruction in the config
            response = model.generate_content(
                contents=[system_instruction + user_message],
            )

            # 5. Extract the generated text content
            bot_message = response.text

            return JsonResponse({'response': bot_message})

        except Exception as e:
            # Catch all exceptions and provide a helpful server error message
            return JsonResponse({'error': f'An internal server error occurred: {str(e)}'}, status=500)

    return JsonResponse({'error': 'Invalid request method. Only POST is allowed.'}, status=400)