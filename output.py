import tensorflow as tf

#### TRANSFORM OUTPUT
def transform_output(output):
  dis = ['Disease_(vertigo) Paroymsal  Positional Vertigo', 'Disease_AIDS', 'Disease_Acne', 'Disease_Alcoholic hepatitis', 'Disease_Allergy', 'Disease_Arthritis', 'Disease_Bronchial Asthma', 'Disease_Cervical spondylosis', 'Disease_Chicken pox', 'Disease_Chronic cholestasis', 'Disease_Common Cold', 'Disease_Dengue', 'Disease_Diabetes ', 'Disease_Dimorphic hemmorhoids(piles)', 'Disease_Drug Reaction', 'Disease_Fungal infection', 'Disease_GERD', 'Disease_Gastroenteritis', 'Disease_Heart attack', 'Disease_Hepatitis B', 'Disease_Hepatitis C', 'Disease_Hepatitis D', 'Disease_Hepatitis E', 'Disease_Hypertension ', 'Disease_Hyperthyroidism', 'Disease_Hypoglycemia', 'Disease_Hypothyroidism', 'Disease_Impetigo', 'Disease_Jaundice', 'Disease_Malaria', 'Disease_Migraine', 'Disease_Osteoarthristis', 'Disease_Paralysis (brain hemorrhage)', 'Disease_Peptic ulcer diseae', 'Disease_Pneumonia', 'Disease_Psoriasis', 'Disease_Tuberculosis', 'Disease_Typhoid', 'Disease_Urinary tract infection', 'Disease_Varicose veins', 'Disease_hepatitis A']
  max_index = tf.argmax(output)
  disease = dis[max_index]
  disease = disease.replace('Disease_','')
  return disease