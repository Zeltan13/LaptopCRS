from openai import OpenAI
import faiss
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

#Initialize the OpenAI client using Your OWN OpenAI API Key please
client = OpenAI(api_key='Your OpenAI API Key')

#Load metadata of 26k Laptops from Amazon Dataset
metadata = pd.read_csv('metadata_cleaned.csv', sep=';')
metadata = metadata.dropna(subset=['title', 'description', 'features'])
#Combine title, description and features into one to create a combined text data for each laptop
metadata['combined_text'] = (
    metadata['title'].astype(str) + ' ' +
    metadata['description'].astype(str) + ' ' +
    metadata['features'].astype(str)
)

#Vectorize the metadata for RAG
texts = metadata['combined_text'].tolist()
print("Vectorizing data...")
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(tqdm(texts)).toarray()

#Build or use FAISS index for RAG (Build it for the first time, use the saved file to use it again so the process will be faster)
if os.path.exists("faiss_index_new.bin"):
    index = faiss.read_index("faiss_index_new.bin")
else:
    index = faiss.IndexFlatL2(vectors.shape[1]) #Create a new FAISS Index
    index.add(np.array(vectors).astype(np.float32)) #Add the vectors to the index
    faiss.write_index(index, "faiss_index_new.bin") #Save the index for future use

#Function to extract laptop specifications using ChatGPT prompt engineering
def extract_specs(user_input, existing_preferences=None):
    #Initialize existing_preferences for a place to store the prefrences that have already been said, and also to add preferences that the user will say
    existing_preferences = existing_preferences or {}
    #Prompt to send to the Fine-Tunend GPT-4o model to extract the specifications from the users input and also to keep track of the existing preferences
    prompt = f"""
    User has provided the following laptop specifications so far: {existing_preferences}.
    The latest input is: "{user_input}".
    Please extract specifications such as brand, budget, RAM, processor, storage, graphics card, and purpose from the latest input.
    Recognize inputs like "gaming laptop" or "for gaming" as the purpose being "gaming."
    Merge the extracted specifications with the existing specs, and return a valid Python dictionary with all keys lowercased.
    Do not include any extra text or explanations.
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

#Function to Retrieve context, aka the best laptops for recommendation depending on the context 
def retrieve_context(query, k=5):
    #Convert the overal query into a TF-IDF vector representation
    query_vector = vectorizer.transform([query]).toarray().astype(np.float32)
    #Perform index search in the FAISS index, distances show us the similarity score, while indices gives us the indexes of the closest laptop matches
    distances, indices = index.search(query_vector, k)
    #Retrieve the data from the indexes found during the FAISS index search, texts[i] provides us the text and distances[0][j] provide us the similarity score
    return [(texts[i], distances[0][j]) for j, i in enumerate(indices[0])]

#Function to format the recommendations so it looks cleaner at the end
def format_recommendation(title, descriptions, specs):
    formatted = f"{title}\n"
    if descriptions:
        formatted += " ".join(descriptions) + " "
    if specs:
        formatted += "Specifications: " + ", ".join(specs) + "."
    return formatted.strip()

#Function for the Combined Model CRS
def recommend_laptop_combined_model():
    preferences = {} #Track the prefrences of the user
    already_asked = set() #Specs that have already been asked or already have been said by the user
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
        print(f"LaptopGPT: Your preferences so far:\n{preferences}") #Show the current preferences captured by the CRS

        missing_specs = [spec for spec in key_specs if spec not in preferences and spec not in already_asked] #Find any missing specifications that have not been collected
        #Finish asking questions if all specifications are covered or the we have sufficient data (5 Laptop Specs)
        if not missing_specs or len(preferences) >= 5:
            break
        #If the CRS hasn't asked the user twice about their specifications (So in total it would be 3 inputs from the user), ask the user for more information 
        if i < 2:
            if missing_specs:
                next_spec = missing_specs[0] #Search for the next missing specification
                already_asked.add(next_spec)  #Mark this specification as it has been asked
                query = query_missing_specs(preferences, next_spec) #Craete a response from the CRS model to seek missing specifications from the user
                print(f"LaptopGPT: {query}") #Print out the response to show to the user
                i += 1

    #Generate the overall query for RAG to search the laptop
    query = " ".join([f"{key}: {value}" for key, value in preferences.items() if value])
    #Retrieve the best laptops, prefereably more than needed so that the Fine-Tuned GPT-4o model can choose the best laptops that the RAG model offers
    rag_results = retrieve_context(query, k=30)
    
    #Use GPT-4o to take the Top-N Laptops with recommendation reasoning for each laptops, first we combine the rag_results into one rag_texts
    rag_texts = "\n".join([text for text, _ in rag_results])
    #Then we create a prompt to guide the Fine-Tuned GPT-4o Moel to generate the Top-N laptops with reasoning for each laptop based on the users preferences
    prompt = f"""
    Based on the following user preferences: {preferences},
    and the retrieved results from the database:
    {rag_texts}
    Provide a ranked list of exactly 5 laptop recommendations.
    For each recommendation, include:
    - Laptop title
    - Specifications (RAM, processor, storage, etc.)
    - Reasoning: Why this laptop is suitable based on the user's preferences.
    """
    #Send a request to Fine-Tuned GPT-4o model to run the prompt
    response = client.chat.completions.create(
        model="ft:gpt-4o-2024-08-06:personal::AW1N4XJq",
        messages=[
            {"role": "system", "content": "You are an expert laptop advisor providing recommendations based on retrieval results."},
            {"role": "user", "content": prompt},
        ],
    )
    #Extract the models response and print it to show to the user the Top-N recommended Laptops for them based on their specifications and needs
    print("LaptopGPT: Here are my top recommendations for you:\n")
    print(response.choices[0].message.content.strip())

recommend_laptop_combined_model()