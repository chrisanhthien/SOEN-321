import ast
import re
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
tqdm.pandas()

# ------------------------------
# LOAD DATA
# ------------------------------
df = pd.read_csv("tiktok_medical_data_cleaned.csv")

# Ensure comments column is usable
df["comments"] = df["comments"].fillna("[]")

# ------------------------------
# FLAGGING TERM LISTS
# ------------------------------
# health terms combine hashtags and text terms
# these terms are taken from hashtags in the challenges column in addition
# to general health-related terms which may appear in  text
health_terms = ['11weekspregnant', '12weekspregnant', '13weekspregnant', '14weekspregnant', '15weekspregnant', '27weekspregnant', '7monthspregnant', '8weekspregnant', 'actuallyautistic', 'acutepain', 'adhd', 'adhdinwomen', 'adhdproblems', 'adhdsymptoms', 'adhdtiktok', 'adhdtok', 'aids', 'aidsawareness', 'aliciakeysbraids', 'anxiety', 'anxietyawareness', 'anxietyhelp', 'anxietyhelper', 'anxietyrelief', 'anxietysquad', 'anxietytips', 'archiveswithzy', 'arthritispainrelief', 'astigmatism', 'astrocytomacanceroustumor', 'austintx', 'autism', 'autismawareness', 'autismmomlife', 'autismunderstanding', 'autistic', 'autisticadult', 'autisticwomenoftiktok', 'backpain', 'backpainexercises', 'backpainrelief', 'beatcancer', 'bestdadever', 'bloodcancer', 'braincancer', 'braincancerawareness', 'braincancerawarenessmonth', 'braincancerfighter', 'braincancersymptoms', 'braintumor', 'breakthestigma', 'breastcancer', 'breastcancerawareness', 'breastcancerawarenessmonth', 'breastcancerin30s', 'breastcancersucks', 'breastcancersymptom', 'breastpain', 'cancer', 'cancerawareness', 'cancerdeestomago', 'cancerdemama', 'cancerdiagnosis', 'cancerfighter', 'cancerfree', 'cancermom', 'cancerpain', 'cancerpatient', 'cancerprevention', 'cancerqueen', 'cancerreconstructivesurgery', 'cancerresearch', 'cancerscreening', 'cancerspread', 'cancerstory', 'cancersucks', 'cancersucks\ud83c\udf97', 'cancersurgery', 'cancersurv', 'cancersurvivor', 'cancersurvivors', 'cancersymptoms', 'cancertiktok', 'cancertok', 'cancertreatment', 'cancerwarrior', 'catdiabetes', 'cervicalcancer', 'cervicalcancerawareness', 'chemorash', 'childhoodcancer', 'childhoodcancerawareness', 'christian', 'christianity', 'christiantiktok', 'chronicallyill', 'chronicdisease', 'chronicillness', 'chronicillnessawareness', 'chronicillnessgrief', 'chronicillnesstiktok', 'chronicinflammation', 'chronicpain', 'chronicpainawareness', 'chronicpainlife', 'chronicpainmanagement', 'chronicpainrelief', 'chronicpainsurvivor', 'chronicpaintiktok', 'chronicpainwarrior', 'chronicpainwarriors', 'chronicwound', 'coloncancer', 'coloncancerawareness', 'coloncancersymptoms', 'colorectacancer', 'colorectcancer', 'complexptsd', 'congestiveheartfailure', 'consejosparadiabetes', 'constipation', 'constipationrelief', 'costofcancer', 'covid', 'covid19', 'covidgov', 'cptsd', 'curecancer', 'depression', 'depressionanxiety', 'diabetes', 'diabetesawareness', 'diabetescheck', 'diabetescontrolada', 'diabetesinfantil', 'diabetesketoacidosis', 'diabetesmanagement', 'diabetesprevention', 'diabetestipo1', 'diabetestipo2', 'diabetestok', 'diabetestype2', 'diostieneelcontrol', 'dontpunishpain', 'emstiktok', 'endaids', 'facepain', 'fcancer', 'fckcancer', 'felinediabetes', 'fightingdiabetes', 'firsttrimesterpregnancy', 'fkcancer', 'flu', 'fluoroquinoloneantibiotics', 'fluoroquinolonetoxicity', 'flushot', 'fuccancer', 'fuckcancer', 'fukcancer', 'gastriccancer', 'gastriccancerstage4', 'gestationaldiabetes', 'godisbiggerthancancer', 'greenscreensticker', 'gyncancers', 'herpesvirus', 'hiv', 'hivawareness', 'hivprevention', 'holistic', 'holisticfertility', 'holistichealing', 'holistichealth', 'holisticjobs', 'holisticpsychiatry', 'holisticpsychology', 'holisticwellness', 'ibstiktok', 'ihatecancer', 'inattentiveadhd', 'infection', 'influencer', 'interesting', 'interestingstorytimes', 'intermittentfasting', 'interventionalpainmanagement', 'intractablepain', 'ivfluids', 'jointpain', 'jugoparadiabetes', 'justiceforjerell', 'katemiddletoncancer', 'kidsfightcancertoo', 'kneepain', 'laborpains', 'latediagnosedautistic', 'latinastiktok', 'latinostiktok', 'legpain', 'lindafightscancer', 'lindasayscancersucks', 'lindascancerrecovery', 'lindascancerupdates', 'livinglifewithcancer', 'livingwithaids', 'longcovid', 'longcovidawareness', 'longtermcovid', 'loveisbiggerthancancer', 'lungcancer', 'lungcancerawareness', 'lungcancerstage3', 'lungcancerstage4', 'lunginfection', 'lvl3autism', 'lymedisease', 'medicalptsd', 'melanomaandskincancerawarenessmonth', 'microinfluencer', 'migraine', 'migrainerelief', 'migraines', 'miscarriageawareness', 'misdiagnosedbipolar', 'mytestimony', 'narcissisticparent', 'neckpain', 'nervepain', 'nervepainrelief', 'nervepainsymptoms', 'nonsmallcelllungcancer', 'noravirus', 'norovirus', 'norovirus2023', 'norovirus2024', 'norovirusoutbreak', 'ocd', 'ocdawareness', 'opioidpainmedicine', 'oralcancer', 'overstimulated', 'overstimulation', 'pain', 'pain when peeing and just in general', 'painadvocate', 'paincommunity', 'painmanagement', 'painmanagementsolutions', 'painmedication', 'painmedicine', 'paintiktok', 'paintok', 'parentsofacancerwarrior', 'pediatriccancer', 'pediatriccancerawareness', 'peleacontraelcancer', 'periodpain', 'pesticides', 'postoppain', 'prediabetes', 'pregnancy', 'pregnancyannouncement', 'pregnancycommunity', 'pregnancycravings', 'pregnancyhumor', 'pregnancylife', 'pregnancyloss', 'pregnancylossawareness', 'pregnancysupport', 'pregnancysymptoms', 'pregnancytiktok', 'pregnancytips', 'pregnancytok', 'pregnant', 'pregnantbelly', 'pregnantcheck', 'pregnantlady', 'pregnantlife', 'pregnantmama', 'pregnantmamasunite', 'pregnantmom', 'pregnanttiktok', 'pregnanttogether', 'procrastinating', 'profoundautism', 'ptsd', 'ptsdawareness', 'ptsdisreal', 'ptsdsurvivor', 'ptsdwarrior', 'questioned', 'questions', 'rarecancer', 'rarediseasestigma', 'rareformlungcancer', 'realisticmomlife', 'recetaparadiabetes', 'remedioparadiabetes', 'reversediabetes', 'ringingthecancerbell', 'saluddiabetes', 'shoulderpain', 'signsandsymptomsofadhd', 'skincancerawareness', 'sobrevivientedecancer', 'spinalcordstimulator', 'stage3breastcancer', 'stage3cancer', 'stage4cancer', 'stage4coloncancer', 'stage4lungcancer', 'std', 'stdawareness', 'sti', 'stickers', 'stillbecoming', 'stitch', 'stomachcancer', 'stomachcancerawareness', 'stomachflu', 'stomachflusucks', 'stomachpain', 'stomachvirus', 'stopdomesticabuse', 'stopdomesticviolence', 'stophivtogether', 'strokeofluck', 't1diabetes', 'terminalcancer', 'testicularcancer', 'thisiscancer', 'thyoidcancer', 'thyroidcancer', 'tipe1diabetes', 'tonguecancer', 'tratamientodiabetes', 'travelanxietytips', 'triplenegativebreastcancer', 'triplepositivebreastcancer', 'trustinggod', 'tumor', 'tumors', 'type1diabetes', 'type2diabetes', 'typeonediabetes', 'uterinecancer', 'uterinecancerawareness', 'vaids', 'vestibular', 'viralinfection', 'viralmed', 'virus', 'viruses', 'walkingpneumonia', 'weirdcovidproblems', 'womenwithadhd', 'youngadultcancer']

health_professional_keywords = ['acceleratednursingprogram', 'adenahealthsystem', 'adrenaldysfunction', 'adrenalfatigue', 'adultchildrenofemotionallyimmatureparents', 'agencynurse', 'alexianbrothersmedicalcenter', 'audreythenurse', 'autismdiagnosis', 'babydoctor', 'babynurse', 'bedridden', 'bedrotting', 'blackdentist', 'blackdentists', 'blackdoctor', 'blackdoctors', 'blackgirldoctor', 'blackregisterednurse', 'bostonnurse', 'breastcancerjourney', 'breastcancerjourney\ud83d\udc97', 'cancerjourney', 'cancernurse', 'celulasmadre', 'childhoodcancerawarness', 'childrenshospital', 'childrenslosangeles', 'chinesehealthpractices', 'chineseherbaldoctor', 'clinic', 'clinical', 'clinicals', 'colorectsurgeon', 'correctionalnurse', 'correctionsnurse', 'cosmeticdentistry', 'criticalcarenurse', 'cvspharmacy', 'dentalburnout', 'dentist', 'dentistry', 'dentistsoftiktok', 'dermdoctor', 'dmd', 'doctor', 'doctoral', 'doctoraldegree', 'doctorarosy', 'doctoravargasguzman', 'doctore', 'doctores', 'doctork', 'doctorlife', 'doctorpatientforum', 'doctors', 'doctorsappointment', 'doctorsbelike', 'doctorsoffice', 'doctorsoftik', 'doctorsoftiktok', 'doctorswife', 'doctorvisit', 'doctorwife', 'dr', 'drained', 'draisahdahlan', 'drallen', 'dramaqueen', 'drasmakhaliq', 'drbenmauck', 'drbryanardis', 'dreadslocks', 'dream', 'dreamsdeferred', 'drgreenv', 'drjarrodbetz', 'drjencaudle', 'drk', 'drkevin', 'drowning', 'drranywoo', 'drsebi', 'drsebiapproved', 'drshaknovsky', 'drsimpson', 'drterrysimpson', 'drydrowning', 'dryu', 'efratlamandre', 'ehlerdanlossyndrome', 'ehlersdanlossyndrome', 'ernurse', 'ernurselife', 'eyedoctor', 'eyedoctorsoftiktok', 'eyedoctortiktok', 'familydentistry', 'femaledentist', 'femaledoctor', 'femaledoctors', 'filipinonurses', 'futuredoctor', 'futurenurse', 'goodnurses', 'guthealthjourney', 'healthadvocate', 'healthblogger', 'healthcare', 'healthcareadministration', 'healthcareawareness', 'healthcareburnout', 'healthcareforalloregon', 'healthcaregirl', 'healthcaregirlie', 'healthcaregirlies', 'healthcaregirls', 'healthcaregirly', 'healthcaregirlyyy', 'healthcareheroes', 'healthcarehumor', 'healthcarelife', 'healthcareprofessional', 'healthcarerealities', 'healthcaretiktok', 'healthcareworker', 'healthcareworkers', 'healthcareworkersbelike', 'healthcareworkersoftiktok', 'healthcoach', 'healthcoachmarci\ud83c\udfcb️\u200d♀️', 'healthconcerns', 'healthjourney', 'healthlivingjourney', 'heartdoctor', 'hillcresthospitaltulsa', 'holisticnursecoach', 'homehealthnurse', 'hormonehealthcoach', 'hospicenursejulie', 'hospicenursesarethebest', 'hospital', 'hospitalbirth', 'hospitalforyoupage', 'hospitalfyp', 'hospitalhumor', 'hospitalized', 'hospitallife', 'hospitallifebelike', 'hospitallifehehe', 'hospitals', 'hospitaltiktoks', 'hospitaltok', 'hypermobileehlersdanlossyndrome', 'hypermobilityspectrumdisorder', 'hypochondriac', 'icunurselife', 'ilooklikeasurgeon', 'imdead', 'justicefornevaeh', 'kiddoctor', 'laboranddeliverynurse', 'linedrawing', 'linedrawings', 'lmd', 'lockinsyndrome', 'lungdoctor', 'malenurse', 'md', 'mdcat', 'medicaljourney', 'medicalschoollearning', 'medschool', 'medschoollife', 'medschoolproblems', 'medstudent', 'medstudentlife', 'medstudentproblems', 'medstudents', 'medstudenttiktok', 'missedhospital', 'mountsinaihospital', 'mycancerjourney', 'nashvillehospital', 'naturopathicdoctor', 'neurosurgeon', 'newgradnurse', 'newnurse', 'newnursesquitting', 'newnursetips', 'nightshiftnurse', 'nurse', 'nurseadvice', 'nurseburnout', 'nursehumor', 'nurselife', 'nurselifebelike', 'nursememes', 'nurseoftiktok', 'nursepractitioner', 'nursepractitioners', 'nursepractitionersoftiktok', 'nurseproblems', 'nurses', 'nursesappreciationweek', 'nursesinspirenurses', 'nurseslife', 'nursesofinstagram', 'nursesoftikok', 'nursesoftiktok', 'nursesrock', 'nursestiktok', 'nursestory', 'nursestorytime', 'nursestudent', 'nursesunite', 'nursesweek', 'nursesweek2025', 'nursetiktok', 'nursetobe', 'nursetok', 'nutritionpsychiatry', 'obgyn', 'oldnurse', 'oncologynurse', 'operatingroomnurse', 'ornurse', 'orthopedicsurgeon', 'oumedicalcenter', 'paindoctor', 'painmanagementphysician', 'painpatient', 'pashsyndrome', 'pcthospital', 'pcthospitaltoks', 'pediatricnurse', 'pharmacist', 'pharmacistsoftiktok', 'pharmacy', 'pharmacytechnician', 'physicaltherapist', 'pinaynurse', 'pinoynurse', 'pmdd', 'pmddawareness', 'postpartumdepression', 'postpartumdoula', 'postpartumjourney', 'posturalorthostatictachycardiasyndrome', 'potssyndrome', 'potssyndromeawareness', 'pregnancyjourney', 'pregnantjourney', 'primarycaredoctor', 'professionalnurse', 'protectourchildren', 'ptsdrecovery', 'quadriplegic', 'redranger', 'redrangerofseattle', 'registerednurse', 'residentdoctor', 'respiratorytherapist', 'respiratorytherapistoftiktok', 'restinparadise', 'rexalldrugs', 'rnnurse', 'roadtodoctor', 'roboticsurgeon', 'sandrafierroactivz', 'seasonednurse', 'silvercrosshospital', 'skindoctor', 'southerncancercenter', 'spinedoctor', 'stalexiushospital', 'stpetershospital', 'studentdoctor', 'studentnurse', 'sugarwarning', 'summermd', 'surgeon', 'surgeons', 'surgeonsoftiktok', 'syndrome', 'thankadoctor', 'thankahealthcareworkertoday', 'thankanurse', 'therapist', 'therapistsontiktok', 'therapistsontiktokt', 'travelnurse', 'travelnurselife', 'tristarcentennialmedicalcenter', 'vanderbiltchildrenshospital', 'waliullahneonatalnurse', 'weightlossdoctor', 'weightlosssurgeon', 'wellnessjourney', 'whitecoatsblackdoctors', 'williamssyndrome', 'winniepalmerhospital', 'withdrawal', 'womendentists', 'womendoctor', 'womenshealthnursepractitioner', 'womensurgeons']

id_terms = [
    'myself', 'i have', 'i suffer',
    'i am diagnosed', 'i was diagnosed', 'i have been diagnosed',
    'was diagnosed with', 'diagnosed with',
    'i got diagnosed', 'i contracted', 'i tested positive', 'i tested negative',
    'years old', 'age is', 'age of', 'birthday', 'dob',
    'my mom', 'my mother', 'my dad', 'my father',
    'my sister', 'my brother', 'my son', 'my daughter',
    'my child', 'my children', 'children', 'child',
    'my wife', 'my husband', 'partner', 'spouse',
    'my grandma', 'my grandfather', 'my grandpa', 'my aunt', 'my uncle',
    'pregnant', 'my pregnancy', 'my baby', 'my newborn'
]

# ------------------------------
# SAFE PARSING OF COMMENTS
# ------------------------------
def parse_comments_safe(text):
    try:
        return ast.literal_eval(text)
    except Exception:
        return []

# ------------------------------
# EXPAND COMMENTS TO LONG FORMAT
# ------------------------------
comment_rows = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Expanding comments"):
    video_id = row.get("video_id", idx)
    challenges = row.get("challenges", "")
    comments_list = parse_comments_safe(row["comments"])

    for c in comments_list:
        if isinstance(c, dict):
            comment_text = c.get("text", "")
        else:
            comment_text = str(c)

        comment_rows.append(
            {
                "video_id": video_id,
                "comment_text": comment_text,
                "challenges": challenges,
            }
        )

comments_df = pd.DataFrame(comment_rows)

# Drop empty comments
comments_df["comment_text"] = comments_df["comment_text"].fillna("").astype(str)
comments_df = comments_df[comments_df["comment_text"].str.strip() != ""].copy()

# ------------------------------
# DISCLOSURE LOGIC
# ------------------------------
def build_keyword_regex(terms):
    escaped = [re.escape(t.strip()) for t in terms if isinstance(t, str) and t.strip() != ""]
    if not escaped:
        return None
    pattern = r"(" + "|".join(escaped) + r")"
    return re.compile(pattern, flags=re.IGNORECASE)

id_regex = build_keyword_regex(id_terms)
health_regex = build_keyword_regex(health_terms)

def is_personal_health_disclosure(text):
    """
    A comment is a personal health disclosure if:
      - It contains at least one id_term
      - AND at least one health term
    """
    if not isinstance(text, str):
        return False

    if id_regex is None or health_regex is None:
        return False

    if not id_regex.search(text):
        return False

    if not health_regex.search(text):
        return False

    return True

comments_df["is_disclosure"] = comments_df["comment_text"].progress_apply(
    is_personal_health_disclosure
)

# ------------------------------
# OVERALL COMMENT-LEVEL METRICS
# ------------------------------
total_comments = len(comments_df)
total_disclosures = comments_df["is_disclosure"].sum()

if total_comments > 0:
    baseline_disclosure_rate = total_disclosures / float(total_comments)
else:
    baseline_disclosure_rate = 0.0

# ------------------------------
# VIDEO-LEVEL METRICS
# ------------------------------
video_disclosure = (
    comments_df.groupby("video_id")["is_disclosure"]
    .any()
    .reset_index(name="video_has_disclosure")
)

total_videos = video_disclosure["video_id"].nunique()
videos_with_disclosure = video_disclosure["video_has_disclosure"].sum()

if total_videos > 0:
    proportion_videos_with_disclosure = videos_with_disclosure / float(total_videos)
else:
    proportion_videos_with_disclosure = 0.0

# ------------------------------
# HASHTAG-LEVEL ANALYSIS
# ------------------------------
# Combine allowed hashtags
allowed_hashtags = set([h.lower() for h in health_terms + health_professional_keywords])

def extract_hashtags_from_challenges(challenges_text):
    if not isinstance(challenges_text, str):
        return []
    parts = re.split(r"[;,]", challenges_text)
    hashtags = []
    for p in parts:
        p_clean = p.strip()
        if p_clean.startswith("#"):
            p_clean = p_clean[1:]
        p_clean = re.sub(r'[\[\]"]', '', p_clean)
        p_clean = p_clean.lower()
        if p_clean != "" and p_clean in allowed_hashtags:
            hashtags.append(p_clean)
    return hashtags

comments_df["hashtags"] = comments_df["challenges"].fillna("").astype(str).apply(
    extract_hashtags_from_challenges
)

comments_df["hashtags"] = comments_df["hashtags"].apply(
    lambda tags: [re.sub(r'[\[\]"]', '', tag) for tag in tags] if isinstance(tags, list) else []
)

comments_exploded = comments_df.explode("hashtags").rename(
    columns={"hashtags": "hashtag"}
)
comments_exploded = comments_exploded[
    comments_exploded["hashtag"].notnull() & (comments_exploded["hashtag"] != "")
    ].copy()

hashtag_stats = (
    comments_exploded.groupby("hashtag")
    .agg(
        total_comments=("comment_text", "count"),
        disclosing_comments=("is_disclosure", "sum"),
    )
    .reset_index()
)

hashtag_stats["disclosure_rate"] = 0.0
mask_nonzero = hashtag_stats["total_comments"] > 0
hashtag_stats.loc[mask_nonzero, "disclosure_rate"] = (
        hashtag_stats.loc[mask_nonzero, "disclosing_comments"]
        / hashtag_stats.loc[mask_nonzero, "total_comments"].astype(float)
)

if baseline_disclosure_rate > 0:
    hashtag_stats["enrichment_factor"] = (
            hashtag_stats["disclosure_rate"] / baseline_disclosure_rate
    )
else:
    hashtag_stats["enrichment_factor"] = 0.0

# ------------------------------
# SAVE RESULTS
# ------------------------------
overall_df = pd.DataFrame(
    {
        "overall_comment_health_disclosure_rate": [baseline_disclosure_rate],
        "proportion_videos_with_health_disclosure_in_comments": [
            proportion_videos_with_disclosure
        ],
        "total_comments": [total_comments],
        "total_disclosing_comments": [int(total_disclosures)],
    }
)

overall_df.to_csv(
    "results_overall_comments_personal_health.csv",
    index=False,
    encoding="utf-8-sig",
)

hashtag_stats.to_csv(
    "results_hashtag_comment_disclosure_rates_all.csv",
    index=False,
    encoding="utf-8-sig",
)
