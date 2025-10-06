import streamlit as st
import pandas as pd
import joblib
import qrcode
from io import BytesIO
from PIL import Image
from pathlib import Path

# Load models
BASE_DIR = Path(__file__).parent
resource_path = BASE_DIR / "resources"
knn = joblib.load(resource_path / "knn_model.pkl")
label_encoder = joblib.load(resource_path / "label_encoder.pkl")
feature_list = joblib.load(resource_path / "feature_columns.pkl")


# Display image
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("broiler.jpg", width=200)

st.title("Broiler Chicken Disease Predictor")

# Prescriptions dictionary
prescriptions = {
    "Newcastle": {
        "English": "No specific treatment. Use oxytetracycline in water.",
        "Kiswahili": "Hakuna matibabu maalum. Tumia oxytetracycline kwenye maji.",
        "Luganda": "Tewali bujjanjabi bwa njawulo. Kozesa oxytetracycline mu mazzi."
    },
    "Coccidiosis": {
        "English": "Use Amprolium, Sulfadimidine, Baycox, or ESB3.",
        "Kiswahili":"Tumia Amprolium, Sulfadimidine, Baycox, au ESB3." ,
        "Luganda":" Kozesa Amprolium, Sulfadimidine, Baycox, oba ESB3."
    },
    "Infectious Bronchitis": {
        "English": "Use enrofloxacin.",
        "Kiswahili":"Tumia enrofloxacin.",
        "Luganda":"Kozesa enrofloxacin."
    },
    "Fowl Cholera": {
        "English": "Use Sulfadimethoxine or tetracycline.",
        "Kiswahili": "Tumia Sulfadimethoxine au tetracycline.",
        "Luganda": "Kozesa Sulfadimethoxine oba tetracycline."
    },
    "Avian Influenza": {
        "English": "No treatment. Isolate and give supportive care.",
        "Kiswahili": " Hakuna matibabu. Kutenga na kutoa huduma ya kuunga mkono",
        "Luganda": "Tewali bujjanjabi. Yawula n'okuwa okulabirira okuwagira."
    },
    "Marek's": {
        "English": "No cure, provide supportive care",
        "Kiswahili": "Hakuna tiba, toa huduma ya kuunga mkono.",
        "Luganda": "Tewali ddagala liwonya, ziwe okulabirira okuwagira."
    },
    "Infectious Coryza": {
        "English": "Use doxycycline or enrofloxacin.",
        "Kiswahili":"Tumia doxycycline au enrofloxacin.",
        "Luganda": "Kozesa doxycycline oba enrofloxacin."
    },
    "Fowl Pox": {
        "English": "Supportive care only.",
        "Kiswahili": "Toa huduma ya kuunga mkono tu.",
        "Luganda": "Ziwe okulabirira okuwagira kwokka."
    },
    "Gumboro (Infectious Bursal Disease)": {
        "English":"Prevent via Gumboro vaccine.",
        "Kiswahili": "Zuia chanjo ya Gumboro.",
        "Luganda": "Ziyiza ngoyita mu kugema obulwadde bwa Gumboro."
    },
    "Brooder Pneumonia": {
        "English": "Use antifungals like Nystatin.",
        "Kiswahili": "Tumia antifungals kama Nystatin.",
        "Luganda": "Kozesa eddagal erita obuwuka nga Nystatin."
    },
    "Salmonellosis": {
        "English" :"Use Enrofloxacin or Amoxicillin.",
        "Kiswahili": "Tumia Enrofloxacin au Amoxicillin.",
        "Luganda": "Kozesa Enrofloxacin oba Amoxicillin."
    },
    "Chronic Respiratory Disease": {
        "English": "Use tylosin or erythromycin.",
        "Kiswahili": "Tumia tylosin au erythromycin.",
        "Luganda": "Kozesa tylosin oba erythromycin."
    }
}

prevention = {
    "Newcastle": {
        "English": "Follow the Vaccination schedules.",
        "Kiswahili": "Fauta ratiba za chanjo.",
        "Luganda": "Goberera enteekateeka z'okugema."
    },
    "Coccidiosis": {
        "English": "Maintain biosecurity and keep sawdust dry and always replace the old sawdust with new",
        "Kiswahili": "Kudumisha biosecurity na uweka kavu ya machungwa na kila wakati ubadilishe sawdust ya zamani na mpya",
        "Luganda": "Kuuma obukuumi n'obukuuta nga bukalu era bulijjo zzawo obukutta obukadde n'ossaamu obupya."
    },
    "Infectious Bronchitis": {
        "English": "Vaccinate as per vaccination schedule and maintain bio-security.",
        "kiswahili": "Chanjo kama kwa ratiba ya chanjo na kudumisha usalama wa bio.",
        "Luganda": "Okugema nga bwe kiri mu nteekateeka y'okugema n'okukuuma obukumi."
    },
    "Fowl Cholera": {
        "English" :"Maintain bio-security.",
        "Kiswahili": "Kudumisha usalama wa bio.",
        "Luganda": "kuuma obukuumi."
    },
    "Avian Influenza":{
        "English" :"Maintain bio-security and hygiene. Also follow the vaccination program.",
        "Kiswahili": "Kudumisha usalama wa bio na usafi wa mazingira. Fauta pia ratiba za chanjo.",
        "Luganda": "okukuuma obukuumi n'obuyonjo. Era goberera enteekateeka z'okugema."
    },
    "Marek's": {
        "English" :"Prevent with vaccination and maintain bio-security.",
        "Kiswahili" : "Kudumisha usalama wa bio. Fauta pia ratiba za chanjo.",
        "Luganda": "okukuuma obukuumi era goberera enteekateeka z'okugema."
    },
    "Infectious Coryza": {
        "English": "Vaccinate as per the vaccination program and maintain bio-security",
        "Kiswahili": "Kudumisha usalama wa bio. Fauta pia ratiba za chanjo.",
        "Luganda": "okukuuma obukuumi era goberera enteekateeka z'okugema."
    },
    "Fowl Pox": {
        "English" :"Follow the vaccination schedules and also control mosquitoes.",
        "Kiswahili": "Fauta ratiba za chanjo na pia udhibiti mbu.",
        "Luganda": "Goberera enteekateeka z'okugema era ofuge ensiri."
    },
    "Gumboro (Infectious Bursal Disease)": {
        "English": "Vaccinate as per the vaccination program and maintain bio-security.",
        "Kiswahili": "Kudumisha usalama wa bio. Fauta pia ratiba za chanjo.",
        "Luganda": "okukuuma obukuumi era goberera enteekateeka z'okugema."
    },
    "Brooder Pneumonia (aspergillosis)": {
        "English" : "Maintain hatchery sanitation, use fresh and dry bedding, feed should be free from mold, maintain good air exchange and avoid over-crowding.",
        "Kiswahili": "Kudumisha usafi wa mazingira, tumbia vumbi safi na kavu, malisho yanapaswa kuwa huru kutoka kwa ukungu, vifaranga wanapaswa kuwa na ubadilishanaji mzuri wa hewa na epuku kufurika",
        "Luganda": "Kuuma obuyonjo, kozesa obukuuta obupya era nga bukalu, emmere erina okubanga nuungi, wabelewo empewo mukiyumba era wewale omujjuzo gwe'enkoko."
    },
    "Salmonellosis": {
        "English" :"Source chicks from salmonella-free hatcheries and maintain good sanitation",
        "Kiswahili": "Chanzo cha vifaranga kutoka kwa hatcheries bure za Salmonella na kudumisha usafi mzuri.",
        "Luganda": "Fuuna o'bukoko okuva mu hatcheries ezitalina salmonella era okuume obuyonjo"
    },
    "Chronic Respiratory Disease": {
        "English" :"Avoid chicks from unknown or backyard sources and maintain bio-security.",
        "Kiswahili": "Epuka vifaranga kutoka kwa vyanzo visivyojulikana na udumishe usalama wa bio",
        "Luganda": "Wewale obukoko okuva mu bifo ebitamanyiddwa era okume obukuumi."
    }
}

interpretation_messages = {
    "low_confidence": {
        "English": "⚠️ Low confidence in top prediction. Please check symptoms or consult a vet.",
        "Kiswahili": "⚠️ Uhakika mdogo katika utabiri huu. Tafadhali hakiki dalili au wasiliana na daktari wa mifugo.",
        "Luganda": "⚠️ Obwesige obutono. Nsaba oddemu okebere obubonero oba weebuze ku musawo w'ebisolo."
    },
    "most_likely": {
        "English": "✅ Most Likely Disease:",
        "Kiswahili": "✅ Ugonjwa Unaowezekana Zaidi:",
        "Luganda": "✅ Obulwadde obusinga okubaamu:"
    },
    "prescription": {
        "English": "💊 Prescription:",
        "Kiswahili": "💊 Maagizo:",
        "Luganda": "💊 Eddagala ly’osobola okukozesa:"
    },
    "prevention": {
        "English": "🛡️ Prevention:",
        "Kiswahili": "🛡️ Kuzuia:",
        "Luganda": "🛡️ Okuziyiza:"
    }
}

# Custom CSS for colors and style
st.markdown("""
    <style>
        body {
            background-color: #f7f9fb;
        }
        .main {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 1rem;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        .stButton button {
            background-color: #FF7F50;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            font-weight: bold;
        }
        .stButton button:hover {
            background-color: #ff5722;
            color: #fff;
        }
    </style>
""", unsafe_allow_html=True)

# App header
st.markdown("<p style='text-align: center; color: grey;'>Get instant prediction and prescription for common broiler diseases</p>", unsafe_allow_html=True)
st.markdown("---")

# Language support
language = st.selectbox("Select Language / Chagua Lugha / Londa Olulimi", ["English", "Kiswahili", "Luganda"])

symptom_translations = {
    "COUGHING": {"English": "Coughing", "Kiswahili": "Kukohoa", "Luganda": "Okukolola"},
    "LABOURED_BREATHING": {"English": "Laboured Breathing", "Kiswahili": "Kutoa pumzi kwa shida", "Luganda": "Okusoomooza mu kukima"},
    "LETHARGY": {"English": "Lethargy", "Kiswahili": "Kulegea", "Luganda": "Obutaba namanyi"},
    "LOSS_OF_APPETITE": {"English": "Loss of Appetite", "Kiswahili": "Kupoteza hamu ya kula", "Luganda": "Obutalya"},
    "SUDDEN_DEATH": {"English": "Sudden Death", "Kiswahili": "Kifo cha ghafla", "Luganda": "Okufa mu bwangu"},
    "RUFFLED_FEATHERS": {"English": "Ruffled Feathers", "Kiswahili": "Manyoya yaliyotatuliwa", "Luganda": "Obwoya obutabangufu"},
    "SNEEZING": {"English": "Sneezing", "Kiswahili": "Kupiga chafya", "Luganda": "Okwasimula"},
    "SKIN_LESIONS": {"English": "Skin Lesions", "Kiswahili": "Vidonda vya ngozi", "Luganda": "Ebizimba ku langi y’omubiri"},
    "DISHARGE_FROM_EYES_&_NOSTRILLS": {"English": "Discharge from Eyes & Nostrils", "Kiswahili": "Uchafu kutoka machoni na puani", "Luganda": "Okufuluma kw’amazzi mu maaso n’akamwa"},
    "WEIGHT_LOSS": {"English": "Weight Loss", "Kiswahili": "Kupungua uzito", "Luganda": "Okwewola obuzito"},
    "LAMENESS": {"English": "Lameness", "Kiswahili": "Kulemewa miguu", "Luganda": "Okulema okutambula"},
    "DEPESSION": {"English": "Depression", "Kiswahili": "Hali ya huzuni", "Luganda": "Obutaba namanyi"},
    "PARALYSIS_OF _WINGS_&_LEGS": {"English": "Paralysis of Wings & Legs", "Kiswahili": "Kupooza mabawa na miguu", "Luganda": "Okusanyalala kwe biwawatilo n’amagulu"},
    "HEAD/NECK_TWISTING": {"English": "Head/Neck Twisting", "Kiswahili": "Kuchezea/kupinda shingo", "Luganda": "Okuzingama kwa mutwe oba ensingo"},
    "HEAD_SHAKING": {"English": "Head Shaking", "Kiswahili": "Kutikisa kichwa", "Luganda": "Okunyenyezesa omutwe"},
    "CONJUNCTIVITIES": {"English": "Conjunctivitis", "Kiswahili": "Kuvimba kwa macho", "Luganda": "Okusiiwa kw’amaaso oba nga mamyufu"},
    "SWOLLEN EYES": {"English": "Swollen Eyes", "Kiswahili": "Macho yaliyovimba", "Luganda": "Amaaso agazimba"},
    "RELUNCTANCE_TO_MOVE": {"English": "Reluctance to Move", "Kiswahili": "Kutotaka kuhamasika", "Luganda": "Obutagala kutambula"},
    "PARALYSIS": {"English": "Paralysis", "Kiswahili": "Kupooza", "Luganda": "Okulemala/okusanyalala"},
    "FEATHERS_LOSS": {"English": "Feathers Loss", "Kiswahili": "Kupoteza manyoya", "Luganda": "Okufiirwa obwoya"}
}

yes_no = {
    "English": ["no", "yes"],
    "Kiswahili": ["hapana", "ndio"],
    "Luganda": ["nedda", "yee"]
}

droppings_label = {
    "English": "Select Droppings Type",
    "Kiswahili": "Chagua Aina ya Kinyesi",
    "Luganda": "Londa langi ya kalimbwe"
}
age_label = {
    "English": "Select Age (weeks)",
    "Kiswahili": "Chagua Umri (wiki)",
    "Luganda": "Londa obukulu (mu wiiki)"
}
predict_label = {
    "English": "🔍 Predict Disease",
    "Kiswahili": "🔍 Tambua Ugonjwa",
    "Luganda": "🔍 Kebela Obulwadde"
}

st.markdown("---")

# Input form
with st.form("prediction_form"):
    age = st.selectbox(age_label[language], list(range(1, 25)))
    droppings = st.selectbox(droppings_label[language], [
        "normal", "bloody diarrhea", "watery diarrhea", 
        "yellow diarrhea", "green diarrhea", "white diarrhea"
    ])

    st.markdown("### " + {
        "English": "Select Symptoms",
        "Kiswahili": "Chagua Dalili",
        "Luganda": "Londa Ebubonero"
    }[language])

    symptom_inputs = {}
    cols = st.columns(2)
    for i, symptom in enumerate(symptom_translations.keys()):
        with cols[i % 2]:
            label = symptom_translations[symptom][language]
            value = st.selectbox(label, yes_no[language], key=symptom)
            symptom_inputs[symptom] = value

    submit_btn = st.form_submit_button(predict_label[language])


# Prediction logic
import numpy as np
import pandas as pd

if submit_btn:
    try:
        # 1. Initialize input vector
        input_data = {feature: 0 for feature in feature_list}

        # 2. Encode age
        age_feature = str(age)
        if age_feature in feature_list:
            input_data[age_feature] = 1
        else:
            st.warning(f"⚠️ Age feature '{age_feature}' not found in model!")

        # 3. Encode droppings
        droppings_feature = droppings.lower()
        if droppings_feature in feature_list:
            input_data[droppings_feature] = 1
        else:
            st.warning(f"⚠️ Droppings feature '{droppings_feature}' not found in model!")

        # 4. Encode symptoms directly
        for sym, val in symptom_inputs.items():
            if val.lower() in ["yes", "ndio", "yee"]:
                if sym in input_data:  
                    input_data[sym] = 1

        # Debugging: show active features
        st.write("✅ Active features:", [k for k,v in input_data.items() if v==1])

        # 5. Check for empty input vector
        if sum(input_data.values()) == 0:
            st.warning("⚠️ No valid features selected. Cannot make a reliable prediction.")
        else:
            # 6. Convert to DataFrame
            df = pd.DataFrame([input_data])
            df = df.reindex(columns=feature_list, fill_value=0).astype(float)

            # 7. Predict probabilities
            probs = knn.predict_proba(df)[0]

            # 8. Get top 3 predictions
            top_indices = probs.argsort()[-3:][::-1]
            top_diseases_encoded = knn.classes_[top_indices]
            top_diseases = label_encoder.inverse_transform(top_diseases_encoded)
            top_diseases = [str(d).strip() for d in top_diseases]

            # 9. Display top predictions
            st.markdown("### 🧠 Top Predicted Diseases with Confidence Scores:")
            for i, idx in enumerate(top_indices):
                disease = top_diseases[i]
                confidence = probs[idx] * 100
                st.markdown(f"**{i+1}. {disease}** — `{confidence:.2f}%`")

            # 10. Handle best disease
            best_disease = top_diseases[0]
            best_confidence = probs[top_indices[0]]
            CONF_THRESHOLD = 0.5
            best_norm = best_disease.strip().lower()

            if best_norm == "healthy":
                st.success({
                    "English": "✅ Your birds are safe from any disease.",
                    "Kiswahili": "✅ Kuku wako wako salama dhidi ya magonjwa.",
                    "Luganda": "✅ Enkoko zo ziri bulungi, tezirina bulwadde."
                }[language])

            elif best_confidence < CONF_THRESHOLD:
                st.warning({
                    "English": "⚠️ No clear disease could be predicted with confidence. Please consult a veterinarian.",
                    "Kiswahili": "⚠️ Hakuna ugonjwa unaoweza kutabiriwa kwa uhakika. Tafadhali wasiliana na daktari wa mifugo.",
                    "Luganda": "⚠️ Tewali bulwadde buzudde na obwesige obw'enjawulo. Nsaba weebuze ku musawo w'ebisolo."
                }[language])

            else:
                st.success(f"{interpretation_messages['most_likely'][language]} **{best_disease}**")

                # Lookup prescription & prevention
                presc_map = {k.strip().lower(): k for k in prescriptions.keys()}
                prev_map = {k.strip().lower(): k for k in prevention.keys()}
                if best_norm in presc_map and best_norm in prev_map:
                    presc_key = presc_map[best_norm]
                    prev_key = prev_map[best_norm]
                    st.markdown(f"{interpretation_messages['prescription'][language]} {prescriptions[presc_key][language]}")
                    st.markdown(f"{interpretation_messages['prevention'][language]} {prevention[prev_key][language]}")
                else:
                    st.warning({
                        "English": f"No prescription/prevention data available for prediction: {best_disease}.",
                        "Kiswahili": f"Haipatikani data ya maagizo kwa utabiri: {best_disease}.",
                        "Luganda": f"Tewali ebiragiro by'eddagala: {best_disease}."
                    }[language])

            # Disclaimer
            st.info({
                "English": "⚠️ If symptoms persist, please consult a veterinarian. This app does not replace professional veterinary services.",
                "Kiswahili": "⚠️ Ikiwa dalili zinaendelea, tafadhali wasiliana na daktari wa mifugo. Programu hii haibadilishi huduma za kitaalamu za mifugo.",
                "Luganda": "⚠️ Singa obubonero busigalawo, nsaba weebuuze ku musawo w'ebisolo. Enkola eno (App) tedda mu kifo kye'empeereza z'abasawo b'ebisolo ez'ekikugu."
            }[language])

            # Confidence visualization
            st.markdown("### 📊 Confidence Visualization")
            confidence_df = pd.DataFrame({
                "Disease": top_diseases,
                "Confidence (%)": [probs[i] * 100 for i in top_indices]
            })
            st.bar_chart(confidence_df.set_index("Disease"))

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")



# QR Code for Sharing
st.markdown("---")
st.markdown("### 📱 Share this App")
app_url = "https://your-streamlit-url.streamlit.app"  

qr = qrcode.make(app_url)
buf = BytesIO()
qr.save(buf)
st.image(Image.open(buf), caption="Scan to open app", width=150)
