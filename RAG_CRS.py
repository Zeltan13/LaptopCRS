from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os

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

#Function to Retrieve context, aka the best laptops for recommendation depending on the context 
def retrieve_context(query, k=5):
    #Convert the overal query into a TF-IDF vector representation
    query_vector = vectorizer.transform([query]).toarray().astype(np.float32)
    #Perform index search in the FAISS index, distances show us the similarity score, while indices gives us the indexes of the closest laptop matches
    distances, indices = index.search(query_vector, k)
    #Retrieve the data from the indexes found during the FAISS index search, texts[i] provides us the text and distances[0][j] provide us the similarity score
    return [(texts[i], distances[0][j]) for j, i in enumerate(indices[0])]

#Function to extract the prefernces from the user input
def extract_preferences(user_input, preferences):
    user_input = user_input.lower() #Make the user input lowercase
    
    #Getting the important specifications of laptop that the user would want from their input
    #Checking if there is a specific brand of laptop the user wants for the laptop
    if "brand" in user_input or any(brand in user_input for brand in ["dell", "lenovo", "hp", "asus", "acer", "apple", "microsoft", "samsung", "msi", "lg", "razer", "huawei"]):
        preferences["brand"] = next((word.capitalize() for word in user_input.split() if word in [
            "dell", "lenovo", "hp", "asus", "acer", "apple", "microsoft", "samsung", "msi", "lg", "razer", "huawei"
        ]), preferences.get("brand"))
    #Checking if there is a specific RAM the user wants for the laptop
    if "ram" in user_input:
        preferences["ram"] = next((word for word in user_input.split() if "gb" in word), preferences.get("ram"))
    #Checking if there is a specific Processor the user wants for the laptop
    if "processor" in user_input or "cpu" in user_input:
        preferences["processor"] = next((word.capitalize() for word in user_input.split() if word in [
            "i3", "i5", "i7", "i9", "ryzen", "amd", "intel", "m1", "m2"
        ]), preferences.get("processor"))
    #Checking if there is a specific type of GPU the user wants for the laptop
    if "gpu" in user_input or "graphics" in user_input:
        preferences["gpu_brand"] = next((word.capitalize() for word in user_input.split() if word in [
            "nvidia", "amd", "intel", "rtx", "gtx", "vega"
        ]), preferences.get("gpu_brand"))
    #Checking if there is a specific storage capacity the user wants for the laptop
    if "storage" in user_input or "hard drive" in user_input:
        preferences["storage_capacity"] = next((word for word in user_input.split() if "gb" in word or "tb" in word), preferences.get("storage_capacity"))
    #Checking if there is a specific ssd that the user has in mind for the laptop
    if "ssd" in user_input or "hdd" in user_input:
        preferences["storage_type"] = "SSD" if "ssd" in user_input else "HDD"
    #Checking if there is a specific budget the user has for the laptop
    if "budget" in user_input or "price" in user_input:
        preferences["price"] = next((word for word in user_input.split() if "$" in word or word.isdigit()), preferences.get("price"))
    #Checking if there is a specific screen size the user wants for the laptop
    if "screen size" in user_input or "display" in user_input:
        words = user_input.split()
        for i, word in enumerate(words):
            if "inch" in word or '"' in word:
                preferences["screen_size"] = word.strip('"').replace("inch", "").strip()
            elif word.isdigit() or word.replace('.', '', 1).isdigit():  
                if i + 1 < len(words) and ("inch" in words[i + 1] or "inches" in words[i + 1]):
                    preferences["screen_size"] = word
    #Checking if there is a specific batterly length the user wants for the laptop
    if "battery" in user_input or "battery life" in user_input:
        if "long" in user_input or "good" in user_input:
            preferences["battery_life"] = "long-lasting"
        else:
            preferences["battery_life"] = next((word for word in user_input.split() if "hour" in word or "hrs" in word), preferences.get("battery_life"))
    #Checking if there is a specific type of GPU the user wants for the laptop
    if "weight" in user_input or "light" in user_input:
        preferences["weight"] = "lightweight" if "light" in user_input else preferences.get("weight")
    #Checking if there is a specific OS the user wants for the laptop
    if "os" in user_input or "operating system" in user_input:
        preferences["os"] = next((word.capitalize() for word in user_input.split() if word in [
            "windows", "macos", "linux", "ubuntu", "chromeos"
        ]), preferences.get("os"))
    #Checking if there is a specific audio quality the user wants for the laptop
    if "audio" in user_input or "sound" in user_input:
        preferences["audio"] = "high-quality audio" if "high-quality" in user_input else preferences.get("audio")
    #Checking if there is a specific type of keyboard the user wants for the laptop
    if "keyboard" in user_input:
        preferences["keyboard_features"] = next((word for word in user_input.split() if word in ["backlit", "rgb"]), preferences.get("keyboard_features"))
    #Checking if there is a specific material the user wants for the laptop
    if "material" in user_input:
        preferences["material"] = next((word.capitalize() for word in user_input.split() if word in [
            "aluminum", "plastic", "carbon"
        ]), preferences.get("material"))
    #Checking if there is a need for a webcam in the laptop, and the quality of webcam
    if "webcam" in user_input or "camera" in user_input:
        if "hd" in user_input or "full hd" in user_input:
            preferences["webcam_quality"] = "HD or Full HD"
        else:
            preferences["webcam_quality"] = preferences.get("webcam_quality")
    #Checking if there is a specific type of connectivity the user needs for the laptop the user wants
    if "connectivity" in user_input or "wifi" in user_input or "bluetooth" in user_input:
        if "wifi 6" in user_input:
            preferences["connectivity"] = "Wi-Fi 6"
        elif "bluetooth" in user_input:
            preferences["connectivity"] = "Bluetooth"
        else:
            preferences["connectivity"] = preferences.get("connectivity")
    #Checking if there is a specific overall purpose for the laptop the user wants
    if "purpose" in user_input or "use" in user_input:
        if "gaming" in user_input:
            preferences["purpose"] = "Gaming"
        elif "work" in user_input:
            preferences["purpose"] = "Work"
        elif "general use" in user_input or "everyday" in user_input:
            preferences["purpose"] = "General Use"
        else:
            preferences["purpose"] = preferences.get("purpose")
    
    return preferences

#Format functions to help give a more human-like response from the system since it doesnt use ChatGPT's LLM for now
def format_preferences(preferences):
    readable_preferences = []
    for i, (key, value) in enumerate(preferences.items(), 1):
        if value:
            readable_preferences.append(f"{i}. {key.replace('_', ' ').capitalize()}: {value}")
    return "\n".join(readable_preferences)

#Function to format the recommendations so it looks cleaner at the end
def format_recommendation(title, descriptions, specs):
    formatted = f"{title}\n"
    if descriptions:
        formatted += " ".join(descriptions) + " "
    if specs:
        formatted += "Specifications: " + ", ".join(specs) + "."
    return formatted.strip()

#Functions to generate the query for the overall RAG to get the specified laptop the user wants
def generate_query(preferences):
    query_parts = []
    if preferences.get("brand"):
        query_parts.append(f"{preferences['brand']} laptop")
    if preferences.get("ram"):
        query_parts.append(f"{preferences['ram']} RAM")
    if preferences.get("processor"):
        query_parts.append(f"{preferences['processor']} processor")
    if preferences.get("gpu_brand"):
        query_parts.append(f"{preferences['gpu_brand']} GPU")
    if preferences.get("storage_capacity") and preferences.get("storage_type"):
        query_parts.append(f"{preferences['storage_capacity']} {preferences['storage_type']}")
    elif preferences.get("storage_capacity"):
        query_parts.append(f"{preferences['storage_capacity']} storage")
    if preferences.get("price"):
        query_parts.append(f"within {preferences['price']} budget")
    if preferences.get("screen_size"):
        query_parts.append(f"{preferences['screen_size']} screen size")
    if preferences.get("battery_life"):
        query_parts.append(f"{preferences['battery_life']} battery life")
    if preferences.get("weight"):
        query_parts.append(f"{preferences['weight']} weight")
    if preferences.get("os"):
        query_parts.append(f"{preferences['os']} operating system")
    if preferences.get("audio"):
        query_parts.append(f"{preferences['audio']} audio")
    if preferences.get("keyboard_features"):
        query_parts.append(f"{preferences['keyboard_features']} keyboard")
    if preferences.get("material"):
        query_parts.append(f"{preferences['material']} material")
    if preferences.get("webcam_quality"):
        query_parts.append(f"{preferences['webcam_quality']} webcam")
    if preferences.get("connectivity"):
        query_parts.append(f"{preferences['connectivity']} connectivity")
    if preferences.get("purpose"):
        query_parts.append(f"for {preferences['purpose']}")
    return " ".join(query_parts)

#Function for the RAG only CRS
def recommend_laptop_rag_only(top_n=5):
    preferences = {} #Track the prefrences of the user
    asked_specs = set()  #Track already asked specifications from the user
    #Key specs to keep track of
    key_specs = [
        "brand", "ram", "processor", "gpu_brand", "storage_capacity", "storage_type",
        "price", "screen_size", "battery_life", "weight", "os", "audio",
        "keyboard_features", "material", "webcam_quality", "connectivity", "purpose"
    ]
    #Flexible prompts for missing specs, so the question asked to the user is not too rigid
    spec_prompts = {
        "brand": [
            "Do you have a particular brand in mind for your laptop? Feel free to mention other specs if you'd like.",
            "Are you leaning towards a specific brand, or do you have other preferences to share?"
        ],
        "ram": [
            "What kind of performance are you looking for? Maybe tell me about RAM or anything else important to you.",
            "Do you have a preference for RAM size or other specifications that matter to you?"
        ],
        "processor": [
            "What type of tasks will you be performing on the laptop? This might help determine the right processor and other specs.",
            "Tell me about the performance you need. Any thoughts on the processor or related features?"
        ],
        "gpu_brand": [
            "Are you planning to use the laptop for gaming, video editing, or something else? Let me know if a specific GPU or other features matter.",
            "What graphics capabilities do you need? Feel free to share any other important specs too."
        ],
        "storage_capacity": [
            "How much storage would be enough for your files and apps? Or let me know if other specs are on your mind.",
            "What are your thoughts on storage size? Anything else you'd like your laptop to have?"
        ],
        "storage_type": [
            "Do you prefer a faster SSD or a larger HDD? Or are there other features you're prioritizing?",
            "What type of storage do you think fits your needs? Feel free to include other specs if you'd like."
        ],
        "price": [
            "What budget range are you thinking about? If there are other key specs you'd like, let me know.",
            "How much are you planning to spend? You can also share other preferences if you'd like."
        ],
        "screen_size": [
            "Do you have a preferred screen size or any other display features you're considering?",
            "What screen size works for you? Or is there something else you'd like your laptop to have?"
        ],
        "battery_life": [
            "Will you need long battery life for travel or work? Let me know if other specs are important too.",
            "How important is battery life to you? Feel free to mention other features you'd like."
        ],
        "weight": [
            "Are you looking for a lightweight option for portability? Any other specs you have in mind?",
            "Do you prefer a lighter laptop? Let me know if there are other features you're considering."
        ],
        "os": [
            "What operating system do you prefer? Or let me know about other features you're prioritizing.",
            "Would you like a specific OS, like Windows or macOS? Any other key specs you'd like?"
        ],
        "audio": [
            "Do you care about high-quality audio for music or video calls? Or are there other features on your mind?",
            "How important is audio quality to you? Let me know if there are other things you're considering."
        ],
        "keyboard_features": [
            "Do you need a backlit keyboard or anything special? Feel free to mention other specs too.",
            "What are your thoughts on keyboard features? You can also tell me about other priorities you have."
        ],
        "material": [
            "Would you prefer a premium build like aluminum or something else? Or are there other specs you'd like?",
            "What kind of build material do you prefer? Feel free to mention any other features too."
        ],
        "webcam_quality": [
            "Will you be using the webcam often? Let me know if there's a quality level or other spec you need.",
            "Do you care about webcam quality? Or is there something else you'd like your laptop to have?"
        ],
        "connectivity": [
            "Do you need any specific connectivity options, like Wi-Fi 6 or Bluetooth? Let me know if there’s more on your mind.",
            "What connectivity features are important to you? Feel free to mention other key specs too."
        ],
        "purpose": [
            "What will you primarily use the laptop for? Feel free to include other preferences as well.",
            "Is this laptop for work, gaming, or general use? Let me know if there are other features you’re considering."
        ]
    }
    #Start the conversation
    print("LaptopGPT: Hello! I'm your laptop advisor.")
    print("LaptopGPT: Tell me what you're looking for in a laptop, like brand, budget, RAM, or purpose.")
    
    for _ in range(3):  #3 interactions between user and LaptopGPT
        user_input = input("User: ") #Get the user input
        preferences = extract_preferences(user_input, preferences) #Extract the prefrences
        
        #Display preferences in a numbered list
        formatted_preferences = format_preferences(preferences) #Show the preferences in a formatted way
        print(f"LaptopGPT: Got it! Your preferences so far:\n{formatted_preferences}") #Show the user the current preferences the CRS has collected
        
        #Check for missing key specs to ask the user if they have not given it so far
        missing_specs = [
            spec for spec in key_specs 
            if (spec not in preferences or preferences[spec] is None) and spec not in asked_specs
        ]
        #Ask the missing specs to the user
        if missing_specs:
            random_spec = random.choice(missing_specs)
            asked_specs.add(random_spec)  #Mark this spec as asked
            random_prompt = random.choice(spec_prompts.get(random_spec, ["Could you provide more details about this?"]))
            print(f"LaptopGPT: {random_prompt}")
        else:
            print("LaptopGPT: Let me find recommendations based on your current preferences.")
            break
        
        #If enough details are collected (5 or more, this can be adjusted based on how specific we want the laptops recommended to the user be), give out a recommendation
        if len(preferences) >= 5:
            print("LaptopGPT: I have enough details to make a recommendation!")
            break

    #Build query and retrieve context using FAISS
    query = generate_query(preferences) #Build a query
    results = retrieve_context(query, k=top_n * 2)  #Retrieve more results to account for invalid ones
    #Filter out entries that have insufficient information
    filtered_results = [
        (text, score) for text, score in results 
        if text.strip() and text != "0 [] []" and "[]" not in text
    ][:top_n]
    #Clean the recommendations into a descriptive format so it is clearer for the user to see
    recommendations = []
    for result, score in filtered_results:
        try:
            title, descriptions, specs = result.split("['", 2)
            descriptions = descriptions.rstrip("']").split("', '") if descriptions else []
            specs = specs.rstrip("']").split("', '") if specs else []
        except ValueError:
            title, descriptions, specs = result, [], []
        recommendations.append((format_recommendation(title.strip(), descriptions, specs), score))
    #Check if fewer results were found than requested, if so explain to the user that there were less laptops found than usual
    if len(filtered_results) < top_n:
        print("LaptopGPT: Sorry, I couldn't find enough matches. Here's what I found so far:")
    #Generate reasoning for recommendations using the users preferences
    if preferences:
        reasoning = "Based on your preferences: " + ", ".join(
            [f"{key.replace('_', ' ')}: {value}" for key, value in preferences.items() if value]
        )
    else:
        print("LaptopGPT: Here are some laptops you might like:")
    #Show the recommendations to the users
    if recommendations:
        print(f"LaptopGPT: Here are the Top-{top_n} laptops for you:")
        for i, (text, _) in enumerate(recommendations):
            print(f"{i + 1}. {text}")
    else:
        print("LaptopGPT: Sorry, I couldn't find any matches. Try providing more details or adjusting your preferences.")

recommend_laptop_rag_only()