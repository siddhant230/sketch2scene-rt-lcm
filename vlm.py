import json
import google.generativeai as genai


class LLMUtility:
    def __init__(self, api_key_path="api_key.json",
                 model_name='gemini-1.5-flash'):
        if api_key_path.endswith(".json"):
            self.api_key = json.load(open(api_key_path))["gemini"]
        else:
            self.api_key = api_key_path

        genai.configure(api_key=self.api_key)
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(m.name)
        self.llm = genai.GenerativeModel(model_name)
        self.prompt = f"""
                  rate this drawing out of 10, also give good and bad features in 2 points each, what further could be improved
                  follow below mentioned format

                  RATING :
                  LOOKS LIKE : 
                  FEATURE ANALYSIS : 
                  """

    def get_rating(self, response):
        score = response.split('\n')[0]
        rating = score.split(":")[-1].split("/")[0].strip()
        return rating

    def get_response_score(self, image):
        response = self.llm.generate_content([self.prompt, image]).text
        rating = self.get_rating(response)
        return response, rating
