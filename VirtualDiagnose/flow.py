import os
from crewai import Agent, Task, Crew, LLM
from crewai_tools import tool
from langchain_community.tools import DuckDuckGoSearchRun


llm = LLM(model='ollama/openhermes',
          base_url="http://localhost:11434",)

@tool("DuckDuckGoSearch")
def search(search_query: str):
    """Search the web for information on a given topic."""
    return DuckDuckGoSearchRun().run(search_query)

doctor = Agent(
    llm = llm,
    role = "Diagnostician",
    goal = "Accurately diagnose a patient's condition",
    backstory = "You are an experienced doctor and diagnostician who helps diagnose a patient's possible conditions based on various factors.",
    allow_delegation = False,
)

# Get symptoms from user
symptoms = input("Enter symptoms: ").strip()
if not symptoms:
    print("No symptoms provided. Please try again.")
    exit()

# Define the task
diagnose_task = Task(
    description = 
    f"""Search the internet to accurately identify a patient's medical condition by analyzing symptoms, 
    medical history, and diagnostic tests. 
    Your goal is to provide a precise diagnosis, including the name of the illness, a detailed description, 
    associated symptoms, treatment options, expected recovery times, and any necessary follow-up care. 
    You will use clinical expertise and critical thinking to guide patients and other healthcare professionals toward 
    effective management and resolution of the condition.
    
    The patient's symptoms are the following: {symptoms}""",
    expected_output =
    f"""
    Based on the symptoms you’ve described — dizziness, a low-grade fever, and a runny nose — the condition could likely be viral upper respiratory tract infection (common cold).

    Diagnosis:
        Name: Viral Upper Respiratory Tract Infection (Common Cold)
        Description: A mild and self-limiting viral infection primarily affecting the nose, throat, and sinuses. Commonly caused by rhinoviruses but can also be due to other viruses like coronaviruses or adenoviruses.

    Symptoms:
        Dizziness (likely from dehydration or sinus pressure)
        Low-grade fever (typically under 100.4°F or 38°C)
        Runny nose (nasal discharge due to inflammation)
        Possible additional symptoms: sore throat, mild cough, fatigue, and nasal congestion
    
    Treatment:
        Hydration: Drink plenty of fluids to prevent dehydration, which may help with dizziness.
        Rest: Ensure adequate rest to support recovery.
        Over-the-counter medications:
            Decongestants (e.g., pseudoephedrine) for nasal congestion.
            Fever reducers (e.g., acetaminophen or ibuprofen) for fever and discomfort.
            Antihistamines (e.g., diphenhydramine) for runny nose relief.
        Steam inhalation or a humidifier to ease nasal passage swelling.
    
    Recovery Time:
        Symptoms generally resolve within 7–10 days. Dizziness and fever may subside earlier with proper care.
    
    Follow-Up:
        If symptoms worsen (e.g., fever exceeds 102°F, severe headache, or difficulty breathing), this may indicate a secondary infection like bacterial sinusitis or another condition requiring further medical attention.
    """,
    agent = doctor,
    tools = [search],
    output_file = "diagnosis.txt"
)


pharmacist = Agent(
    llm = llm,
    role = "Pharmacist",
    goal = "Accurately provide medication recommendations for a patient's condition",
    backstory = "You are an experienced doctor and pharmacist who helps provide medication recommendations for a patient's condition based on the doctor's diagnosis.",
    allow_delegation = False,
)

# Get allergies from user
allergies = input("Any medical allergies: ").strip()
if not allergies:
    print("No response provided. Please try again.")
    exit()

# Define the task
meds_recommendation_task = Task(
    description = 
    f"""Search the internet to accurately provide medication recommendations for a patient's medical condition 
    by analyzing symptoms, medical history, and diagnosis from the doctor. 
    Your goal is to provide a precise recommendation of medication(s) (including prescripted medications and over-the-counter medications), including the name of the medication(s), a detailed description 
    of the medicine(s), whether the medicine(s) are prescripted, instructions to take the medicine(s), any side effects and precautions, expected recovery times, and any necessary follow-up care. 
    You will use clinical expertise and critical thinking to guide patients toward effective management and resolution of the condition. 
    Make sure to take into account the patient's allergies: {allergies}""",
    expected_output =
    f"""
    Medication Recommendation for Common Cold:
    Medications:
    1. Acetaminophen (Paracetamol):
        - Description: A widely used over-the-counter medication for reducing fever and relieving mild to moderate pain.
        - Instructions: Take 500–1000 mg every 4–6 hours as needed for fever or pain, but do not exceed 4000 mg in 24 hours.
        - Side Effects: Rare but may include nausea, rash, or liver damage if overdosed.
        - Precautions: Avoid alcohol while taking this medication to reduce liver strain.
    2. Pseudoephedrine (Sudafed):
        - Description: A decongestant that reduces nasal congestion by shrinking swollen blood vessels in the nasal passages.
        - Instructions: Take 60 mg every 4–6 hours, not exceeding 240 mg per day.
        - Side Effects: Insomnia, nervousness, or dizziness.
        - Precautions: Not recommended for people with high blood pressure or heart disease. Use with caution.
    3. Diphenhydramine (Benadryl):
        - Description: An antihistamine used to reduce runny nose and help with better sleep by alleviating nighttime symptoms.
        - Instructions: Take 25–50 mg every 6–8 hours as needed for symptoms, especially at bedtime.
        - Side Effects: Drowsiness, dry mouth, or dizziness.
        - Precautions: Avoid driving or operating machinery due to sedative effects.
    4. Honey and Lemon (Natural Remedy):
        - Description: A soothing mixture for sore throat and cough.
        - Instructions: Mix 1–2 tablespoons of honey with freshly squeezed lemon juice in warm water. Drink as needed.
        - Side Effects: None, but honey should not be given to children under 1 year old.
        - Precautions: Ensure the honey is pure and not adulterated.
    
    Expected Recovery Time:
    Most symptoms improve within 7–10 days. Rest, hydration, and the medications above should help accelerate recovery.
    
    Follow-Up Care:
    If symptoms persist beyond 10 days or worsen (e.g., severe headache, shortness of breath, fever above 102°F), consult a healthcare provider. This could indicate a secondary bacterial infection, such as sinusitis or pneumonia.
    """,
    agent = pharmacist,
    # tools = [search],
    output_file = "meds.txt"
)


product_analyst = Agent(
    llm = llm,
    role = "Product Analyst",
    goal = "Accurately provide pricing, effectiveness, side effects, etc for the medications recommended by the pharmacist",
    backstory = 
    """You are an experienced doctor, pharmacist, and product analyst who helps to provide medication 
    product information based on the pharmacist's medication recommendations.""",
    allow_delegation = False,
)

# Define the task
meds_product_task = Task(
    description = 
    f"""Search the internet to accurately provide medication product information for a patient's medication recommendation
    by the pharamacist. 
    Your goal is to provide a precise list of medication products, including the name of the medications, the brand of the product, 
    a detailed description of the product, price of the product, any side effects and precautions, whether it is over-the-counter or prescripted, as well as a website url for the product.
    Make sure to rank the products by taking an overall consideration of the price, brand reliability and legitamacy, and side effects of the products.""",
    expected_output =
    f"""
    Medication Recommendation for Common Cold
    1. Tylenol Extra Strength (Acetaminophen)
        - Brand: Tylenol
        - Description: A trusted over-the-counter pain reliever and fever reducer effective for mild to moderate symptoms like headache and fever.
        - Price: $8–$12 for a 24-count pack (500 mg tablets).
        - Side Effects: Rare, but may include nausea, rash, or liver damage if overdosed.
        - Precautions: Avoid alcohol and do not exceed 4000 mg in 24 hours.
        - Reliability: Highly reputable brand with consistent quality and efficacy.
        - Prescription Status: Not prescribed (Available over-the-counter).
        - Product Link: https://fakeurl.com
    2. Sudafed PE (Pseudoephedrine)
        - Brand: Sudafed
        - Description: A decongestant that effectively alleviates nasal congestion by shrinking blood vessels in nasal passages.
        - Price: $9–$15 for a 36-count pack (10 mg tablets).
        - Side Effects: Insomnia, nervousness, or dizziness.
        - Precautions: Avoid in individuals with high blood pressure or heart disease. Use with caution if pregnant.
        - Reliability: Market leader for decongestants; widely recommended by healthcare professionals.
        - Prescription Status: Not prescribed (Available over-the-counter; sometimes restricted due to pseudoephedrine content).
        - Product Link: Buy Sudafed PE on Amazon
    3. Benadryl Allergy Plus Congestion (Diphenhydramine + Phenylephrine)
        - Brand: Benadryl
        - Description: Combines an antihistamine for runny nose and sneezing with a decongestant for congestion relief.
        - Price: $10–$14 for a 24-count pack.
        - Side Effects: Drowsiness, dry mouth, or dizziness.
        - Precautions: May cause significant sedation; not recommended for daytime use if operating machinery or driving.
        - Reliability: Well-established brand with dual-action relief for nighttime cold symptoms.
        - Prescription Status: Not prescribed (Available over-the-counter).
        - Product Link: Buy Benadryl Allergy Plus Congestion on Amazon
    4. Mucinex DM (Guaifenesin + Dextromethorphan)
        - Brand: Mucinex
        - Description: An expectorant and cough suppressant that reduces mucus and soothes persistent coughs.
        - Price: $15–$20 for a 20-count pack (12-hour extended-release tablets).
        - Side Effects: Upset stomach, dizziness, or headache.
        - Precautions: Drink plenty of fluids for optimal mucus-thinning effects. Avoid with MAO inhibitors.
        - Reliability: Highly rated for respiratory symptom management.
        - Prescription Status: Not prescribed (Available over-the-counter).
        - Product Link: Buy Mucinex DM on Amazon
    5. NyQuil Cold & Flu (Acetaminophen, Dextromethorphan, and Doxylamine)
        - Brand: NyQuil
        - Description: A multi-symptom relief medication for fever, aches, cough, and nasal congestion. Includes a sedating antihistamine for restful sleep.
        - Price: $12–$18 for a 12 oz bottle (liquid form) or $10–$15 for 16 LiquiCaps.
        - Side Effects: Drowsiness, dry mouth, or upset stomach.
        - Precautions: Avoid alcohol; do not use during the daytime or with other acetaminophen products.
        - Reliability: Trusted for nighttime relief; highly popular for its multi-symptom efficacy.
        - Prescription Status: Not prescribed (Available over-the-counter).
        - Product Link: Buy NyQuil Cold & Flu on Amazon
    
    Ranking Considerations:
    1. Tylenol Extra Strength: Best for fever and mild pain; reliable and affordable.
    2. Sudafed PE: Best for congestion with minimal side effects in healthy adults.
    3. Benadryl Allergy Plus Congestion: Ideal for nighttime use with effective symptom relief.
    4. Mucinex DM: Great for persistent cough and chest congestion relief.
    5. NyQuil Cold & Flu: Best for comprehensive nighttime symptom management.
    """,
    agent = product_analyst,
    tools = [search],
    output_file = "products.txt"
)

crew = Crew(agents=[doctor, pharmacist, product_analyst], tasks=[diagnose_task, meds_recommendation_task, meds_product_task], verbose=True)

try:
    task_output = crew.kickoff()
    print(task_output)
except Exception as e:
    print(f"Error during task execution: {e}")