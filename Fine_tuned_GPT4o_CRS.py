from openai import OpenAI
import ast

#Initialize the OpenAI client using Your OWN OpenAI API Key please
client = OpenAI(api_key='Your OpenAI API Key')

#Function to extract laptop specifications using ChatGPT prompt engineering
def extract_specs(user_input, existing_preferences=None):
    #Initialize existing_preferences for a place to store the prefrences that have already been said, and also to add preferences that the user will say
    existing_preferences = existing_preferences or {}
    #Prompt to send to the Fine-Tunend GPT-4o model to extract the specifications from the users input and also to keep track of the existing preferences
    prompt = f"""
    User has provided the following laptop specifications so far: {existing_preferences}.
    The latest input is: "{user_input}".
    Please extract specifications such as brand, budget, RAM, processor, storage, graphics card, and purpose from the latest input, and merge them with the existing specs.
    Provide the response as a Python dictionary.
    """
    #Send a request to Fine-Tuned GPT-4o model to run the prompt
    response = client.chat.completions.create(
        model="ft:gpt-4o-2024-08-06:personal::AW1N4XJq", #Use the Fine-Tuned GPT-4o Model that we have trained to recommend laptops
        messages=[
            {"role": "system", "content": "You are a knowledgeable assistant extracting laptop specifications from user input."},
            {"role": "user", "content": prompt},
        ],
    )
    #Parse the response into a Python dictionary, handle errors by returning the existing prefrences
    try:
        specs_dict = ast.literal_eval(response.choices[0].message.content.strip())
        return specs_dict
    except Exception as e:
        print("Error parsing response:", e)
        return existing_preferences

#Function to create a response from Fine-Tuned GPT-4o Model to seek missing specifications from the user
def query_missing_specs(preferences, missing_specs):
    #Prompt to guide the Fine-Tuned GPT-4o Model to generate a natural sounding response to seek missing specifications from the user
    prompt = f"""
    User's current preferences are: {preferences}.
    The assistant needs to ask the user about their {missing_specs}.
    Generate a natural-sounding query for this.
    """
    #Send a request to Fine-Tuned GPT-4o model to run the prompt
    response = client.chat.completions.create(
        model="ft:gpt-4o-2024-08-06:personal::AW1N4XJq",
        messages=[
            {"role": "system", "content": "You are an assistant designed to ask users for missing specifications in a natural tone."},
            {"role": "user", "content": prompt},
        ],
    )
    #Extract the models response and return it as a string
    return response.choices[0].message.content.strip()

#Function to format the user's prefrences into a numbered list for readability
def format_preferences(preferences):
    #Clean up the preferences into a formatted string where each preference is displayed as "1. Key: Value" until the last preference so the user can see what preferences are captured by the model
    formatted = "\n".join([f"{i + 1}. {key.replace('_', ' ').capitalize()}: {value}" for i, (key, value) in enumerate(preferences.items()) if value])
    #Returns the final formatted preferences
    return f"Here are your preferences so far:\n{formatted}"

#Function to recommend a ranked list of the Top-N laptops based on the users' prefereces
def recommend_laptops_top_n(preferences, top_n=5):
    #Prompt to guide the Fine-Tuned GPT-4o Model to generate a list of Top-N laptops based on the users preferencs
    prompt = f"""
    User's preferences: {preferences}.
    Provide a ranked list of the top {top_n} laptop recommendations based on how closely they match the user's preferences.
    Include the laptop details and explain why each one was chosen.
    Allow partial matches and prioritize the closest options.
    """
    #Send a request to Fine-Tuned GPT-4o model to run the prompt
    response = client.chat.completions.create(
        model="ft:gpt-4o-2024-08-06:personal::AW1N4XJq",
        messages=[
            {"role": "system", "content": "You are an expert laptop advisor providing personalized recommendations."},
            {"role": "user", "content": prompt},
        ],
    )
    #Extract the models response into a string to send back to the user
    return response.choices[0].message.content.strip()

#Function for the Fine-Tuned GPT-4o Model Only CRS
def recommend_laptop_fine_tuned_gpt4o_only():
    preferences = {} #Track the prefrences of the user
    #Key specs to keep track of
    key_specs = [
        "brand", "ram", "processor", "gpu_brand", "storage_capacity", "storage_type",
        "price", "screen_size", "battery_life", "weight", "os", "audio",
        "keyboard_features", "material", "webcam_quality", "connectivity", "purpose"
    ]
    #Start the conversation
    print("LaptopGPT: Hello! I'm your laptop advisor.")
    print("LaptopGPT: Tell me what you're looking for in a laptop, like brand, budget, RAM, or purpose.")
    i = 0 #i variable to keep the CRS from querying missing specs one more time than needed
    for _ in range(3): #3 interactions between user and LaptopGPT
        user_input = input("User: ") #Get the user input
        preferences = extract_specs(user_input, preferences) #Extract user preferences from users input
        formatted_preferences = format_preferences(preferences) #Format the preferences 
        print(f"LaptopGPT: {formatted_preferences}") #Show the current preferences captured by the Fine-Tuned GPT-4o Model to the user
        
        missing_specs = [spec for spec in key_specs if spec not in preferences] #Find any missing specifications that have not been collected
        if not missing_specs or len(preferences) >= 5: #If enough specifications are collected (5 or more, this can be adjusted based on how specific we want the laptops recommended to the user be), break from the loop and give out the recommendation
            break
        if i < 2: #If the CRS hasn't asked the user twice about their specifications (So in total it would be 3 inputs from the user), ask the user for more information 
            query = query_missing_specs(preferences, missing_specs[0])  #Craete a response from the CRS model to seek missing specifications from the user
            print(f"LaptopGPT: {query}") #Print out the response to show to the user
            i += 1
    top_n = 5 #Amount of laptops being recommended
    #Generate Top-N recommendations
    recommendations_text = recommend_laptops_top_n(preferences, top_n)
    print(f"LaptopGPT: Here are my top-{top_n} recommendations for you:\n")
    print(recommendations_text)

recommend_laptop_fine_tuned_gpt4o_only()