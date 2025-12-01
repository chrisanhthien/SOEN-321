import re
import ast
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

# -----------------------------------------
# LOAD & CLEAN DATA
# -----------------------------------------

df = pd.read_csv("tiktok_data_medical_large.csv")

# Ensure key text fields are not null
df["desc"] = df["desc"].fillna("")
df["challenges"] = df["challenges"].fillna("[]")

# -----------------------------------------
# FLAGGING TERM LISTS
# -----------------------------------------
# health terms combine hashtags and text terms
# these terms are taken from hashtags in the challenges column in addition
# to general health-related terms which may appear in the description text
health_terms = ['11weekspregnant', '12weekspregnant', '13weekspregnant', '14weekspregnant', '15weekspregnant', '27weekspregnant', '7monthspregnant', '8weekspregnant', 'actuallyautistic', 'acutepain', 'adhd', 'adhdinwomen', 'adhdproblems', 'adhdsymptoms', 'adhdtiktok', 'adhdtok', 'aids', 'aidsawareness', 'aliciakeysbraids', 'anxiety', 'anxietyawareness', 'anxietyhelp', 'anxietyhelper', 'anxietyrelief', 'anxietysquad', 'anxietytips', 'archiveswithzy', 'arthritispainrelief', 'astigmatism', 'astrocytomacanceroustumor', 'austintx', 'autism', 'autismawareness', 'autismmomlife', 'autismunderstanding', 'autistic', 'autisticadult', 'autisticwomenoftiktok', 'backpain', 'backpainexercises', 'backpainrelief', 'beatcancer', 'bestdadever', 'bloodcancer', 'braincancer', 'braincancerawareness', 'braincancerawarenessmonth', 'braincancerfighter', 'braincancersymptoms', 'braintumor', 'breakthestigma', 'breastcancer', 'breastcancerawareness', 'breastcancerawarenessmonth', 'breastcancerin30s', 'breastcancersucks', 'breastcancersymptom', 'breastpain', 'cancer', 'cancerawareness', 'cancerdeestomago', 'cancerdemama', 'cancerdiagnosis', 'cancerfighter', 'cancerfree', 'cancermom', 'cancerpain', 'cancerpatient', 'cancerprevention', 'cancerqueen', 'cancerreconstructivesurgery', 'cancerresearch', 'cancerscreening', 'cancerspread', 'cancerstory', 'cancersucks', 'cancersucks\ud83c\udf97', 'cancersurgery', 'cancersurv', 'cancersurvivor', 'cancersurvivors', 'cancersymptoms', 'cancertiktok', 'cancertok', 'cancertreatment', 'cancerwarrior', 'catdiabetes', 'cervicalcancer', 'cervicalcancerawareness', 'chemorash', 'childhoodcancer', 'childhoodcancerawareness', 'christian', 'christianity', 'christiantiktok', 'chronicallyill', 'chronicdisease', 'chronicillness', 'chronicillnessawareness', 'chronicillnessgrief', 'chronicillnesstiktok', 'chronicinflammation', 'chronicpain', 'chronicpainawareness', 'chronicpainlife', 'chronicpainmanagement', 'chronicpainrelief', 'chronicpainsurvivor', 'chronicpaintiktok', 'chronicpainwarrior', 'chronicpainwarriors', 'chronicwound', 'coloncancer', 'coloncancerawareness', 'coloncancersymptoms', 'colorectacancer', 'colorectcancer', 'complexptsd', 'congestiveheartfailure', 'consejosparadiabetes', 'constipation', 'constipationrelief', 'costofcancer', 'covid', 'covid19', 'covidgov', 'cptsd', 'curecancer', 'depression', 'depressionanxiety', 'diabetes', 'diabetesawareness', 'diabetescheck', 'diabetescontrolada', 'diabetesinfantil', 'diabetesketoacidosis', 'diabetesmanagement', 'diabetesprevention', 'diabetestipo1', 'diabetestipo2', 'diabetestok', 'diabetestype2', 'diostieneelcontrol', 'dontpunishpain', 'emstiktok', 'endaids', 'facepain', 'fcancer', 'fckcancer', 'felinediabetes', 'fightingdiabetes', 'firsttrimesterpregnancy', 'fkcancer', 'flu', 'fluoroquinoloneantibiotics', 'fluoroquinolonetoxicity', 'flushot', 'fuccancer', 'fuckcancer', 'fukcancer', 'gastriccancer', 'gastriccancerstage4', 'gestationaldiabetes', 'godisbiggerthancancer', 'greenscreensticker', 'gyncancers', 'herpesvirus', 'hiv', 'hivawareness', 'hivprevention', 'holistic', 'holisticfertility', 'holistichealing', 'holistichealth', 'holisticjobs', 'holisticpsychiatry', 'holisticpsychology', 'holisticwellness', 'ibstiktok', 'ihatecancer', 'inattentiveadhd', 'infection', 'influencer', 'interesting', 'interestingstorytimes', 'intermittentfasting', 'interventionalpainmanagement', 'intractablepain', 'ivfluids', 'jointpain', 'jugoparadiabetes', 'justiceforjerell', 'katemiddletoncancer', 'kidsfightcancertoo', 'kneepain', 'laborpains', 'latediagnosedautistic', 'latinastiktok', 'latinostiktok', 'legpain', 'lindafightscancer', 'lindasayscancersucks', 'lindascancerrecovery', 'lindascancerupdates', 'livinglifewithcancer', 'livingwithaids', 'longcovid', 'longcovidawareness', 'longtermcovid', 'loveisbiggerthancancer', 'lungcancer', 'lungcancerawareness', 'lungcancerstage3', 'lungcancerstage4', 'lunginfection', 'lvl3autism', 'lymedisease', 'medicalptsd', 'melanomaandskincancerawarenessmonth', 'microinfluencer', 'migraine', 'migrainerelief', 'migraines', 'miscarriageawareness', 'misdiagnosedbipolar', 'mytestimony', 'narcissisticparent', 'neckpain', 'nervepain', 'nervepainrelief', 'nervepainsymptoms', 'nonsmallcelllungcancer', 'noravirus', 'norovirus', 'norovirus2023', 'norovirus2024', 'norovirusoutbreak', 'ocd', 'ocdawareness', 'opioidpainmedicine', 'oralcancer', 'overstimulated', 'overstimulation', 'pain', 'pain when peeing and just in general', 'painadvocate', 'paincommunity', 'painmanagement', 'painmanagementsolutions', 'painmedication', 'painmedicine', 'paintiktok', 'paintok', 'parentsofacancerwarrior', 'pediatriccancer', 'pediatriccancerawareness', 'peleacontraelcancer', 'periodpain', 'pesticides', 'postoppain', 'prediabetes', 'pregnancy', 'pregnancyannouncement', 'pregnancycommunity', 'pregnancycravings', 'pregnancyhumor', 'pregnancylife', 'pregnancyloss', 'pregnancylossawareness', 'pregnancysupport', 'pregnancysymptoms', 'pregnancytiktok', 'pregnancytips', 'pregnancytok', 'pregnant', 'pregnantbelly', 'pregnantcheck', 'pregnantlady', 'pregnantlife', 'pregnantmama', 'pregnantmamasunite', 'pregnantmom', 'pregnanttiktok', 'pregnanttogether', 'procrastinating', 'profoundautism', 'ptsd', 'ptsdawareness', 'ptsdisreal', 'ptsdsurvivor', 'ptsdwarrior', 'questioned', 'questions', 'rarecancer', 'rarediseasestigma', 'rareformlungcancer', 'realisticmomlife', 'recetaparadiabetes', 'remedioparadiabetes', 'reversediabetes', 'ringingthecancerbell', 'saluddiabetes', 'shoulderpain', 'signsandsymptomsofadhd', 'skincancerawareness', 'sobrevivientedecancer', 'spinalcordstimulator', 'stage3breastcancer', 'stage3cancer', 'stage4cancer', 'stage4coloncancer', 'stage4lungcancer', 'std', 'stdawareness', 'sti', 'stickers', 'stillbecoming', 'stitch', 'stomachcancer', 'stomachcancerawareness', 'stomachflu', 'stomachflusucks', 'stomachpain', 'stomachvirus', 'stopdomesticabuse', 'stopdomesticviolence', 'stophivtogether', 'strokeofluck', 't1diabetes', 'terminalcancer', 'testicularcancer', 'thisiscancer', 'thyoidcancer', 'thyroidcancer', 'tipe1diabetes', 'tonguecancer', 'tratamientodiabetes', 'travelanxietytips', 'triplenegativebreastcancer', 'triplepositivebreastcancer', 'trustinggod', 'tumor', 'tumors', 'type1diabetes', 'type2diabetes', 'typeonediabetes', 'uterinecancer', 'uterinecancerawareness', 'vaids', 'vestibular', 'viralinfection', 'viralmed', 'virus', 'viruses', 'walkingpneumonia', 'weirdcovidproblems', 'womenwithadhd', 'youngadultcancer']

# Identification terms (self-identifying language)
identification_terms = [
    "i have", "ive got", "i got", "i was diagnosed",
    "my diagnosis", "i am diagnosed", "i am autistic",
    "i am bipolar", "i am depressed", "i am anxious",
    "i have adhd", "i have autism", "i have ocd",
    "i have ptsd", "i have cancer", "i have diabetes",
    "i have asthma", "i have hiv", "i have aids",
    "my cancer", "my diabetes", "my asthma",
    "my hiv", "my aids", "my disability",
    "living with", "struggling with", "suffering from",
    "i take medication", "i take meds",
    "on meds", "on medication",
    "i am in therapy", "my therapist", "my psychiatrist",
    "in remission", "in recovery", "in treatment",
    "i use a wheelchair", "i use a cane", "i use crutches"
]

# -----------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------

def text_contains_any(text, term_list):
    text_lower = text.lower()
    for term in term_list:
        if term in text_lower:
            return True
    return False

def extract_hashtags_from_text(text):
    hashtags = re.findall(r"#(\w+)", text)
    return hashtags

def parse_challenges(raw):
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return parsed
        return []
    except Exception:
        return []

# -----------------------------------------
# FLAG VIDEOS WITH HEALTH / ID TERMS IN DESCRIPTION
# -----------------------------------------

df["health_terms_in_description"] = df["desc"].progress_apply(
    lambda x: 1 if text_contains_any(x, health_terms) else 0
)

df["identification_in_description"] = df["desc"].progress_apply(
    lambda x: 1 if text_contains_any(x, identification_terms) else 0
)

# Health + identification disclosure flag in description
df["health_and_identification_in_description"] = (
        df["health_terms_in_description"] * df["identification_in_description"]
)

# -----------------------------------------
# EXTRACT HASHTAGS FROM DESCRIPTION TEXT
# -----------------------------------------

df["desc_hashtags"] = df["desc"].progress_apply(extract_hashtags_from_text)

# -----------------------------------------
# HASHTAG ANALYSIS:
# For each hashtag: Of all videos with a health disclosure in the
# description, what fraction contain this hashtag?
# -----------------------------------------

# Filter to videos with health + identification disclosure in description
df_disclose = df[df["health_and_identification_in_description"] == 1].copy()

total_disclosing_videos = len(df_disclose)
print("Total disclosing videos:")
print(total_disclosing_videos)

# Build per-(video, hashtag) rows for disclosing videos
hashtag_rows = []

for _, row in tqdm(df_disclose.iterrows(), total=df_disclose.shape[0]):
    hashtags = row["desc_hashtags"]
    if not hashtags:
        continue

    for ht in set(hashtags):
        ht_lower = ht.lower()
        if ht_lower in health_terms:   # only keep hashtags from health_terms
            hashtag_rows.append({"hashtag": ht_lower})

hashtag_df = pd.DataFrame(hashtag_rows)

print("Head of hashtag_df:")
print(hashtag_df.head())

# Count how many disclosing videos contain each hashtag
hashtag_counts = hashtag_df["hashtag"].value_counts().to_frame("videos_with_hashtag")

# Fraction of all disclosing videos that contain this hashtag
if total_disclosing_videos > 0:
    hashtag_counts["fraction_of_disclosing_videos_with_hashtag"] = (
            hashtag_counts["videos_with_hashtag"] / float(total_disclosing_videos)
    )
else:
    hashtag_counts["fraction_of_disclosing_videos_with_hashtag"] = 0.0

# Sort by highest fraction
hashtag_counts = hashtag_counts.sort_values(
    by="fraction_of_disclosing_videos_with_hashtag",
    ascending=False
)

print("Head of hashtag_counts:")
print(hashtag_counts.head())


# Create safe boundary regex for multi-word terms
health_pattern = rf"(?<!\w)(?:{'|'.join(map(re.escape, health_terms))})(?!\w)"
id_pattern = rf"(?<!\w)(?:{'|'.join(map(re.escape, identification_terms))})(?!\w)"

# Check for health terms in description
df['desc_has_health'] = df['desc'].str.contains(
    health_pattern,
    case=False,
    na=False,
    regex=True
)

# Check for identity terms in description
df['desc_has_id'] = df['desc'].str.contains(
    id_pattern,
    case=False,
    na=False,
    regex=True
)

# Combined flag: description contains at least 1 health term + 1 identification term
df['desc_personal_health_disclosure'] = df['desc_has_health'] & df['desc_has_id']

# Prevalence across dataset
prevalence_health_terms = df['desc_has_health'].mean()
prevalence_id_terms = df['desc_has_id'].mean()
prevalence_combined = df['desc_personal_health_disclosure'].mean()

combined_prevalence_df = pd.DataFrame({
    "prevalence_health_terms_in_description": [prevalence_health_terms],
    "prevalence_identification_terms_in_description": [prevalence_id_terms],
    "prevalence_personal_health_disclosures_in_description": [prevalence_combined]
})


# -----------------------------------------
# SAVE RESULTS
# -----------------------------------------

hashtag_counts.to_csv(
    "results_hashtag_fraction_of_disclosing_videos.csv",
    encoding="utf-8"
)

combined_prevalence_df.to_csv(
    "prevalence_description_all_metrics.csv",
    index=False,
    encoding="utf-8"
)