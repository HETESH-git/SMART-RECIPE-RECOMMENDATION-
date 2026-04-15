import streamlit as st
import pandas as pd
import requests
from rapidfuzz import fuzz
import json
import os
import urllib.parse
from typer import prompt
# ===========================
# Helper links for ChefBot
# ===========================
def recipe_link(dish_name):
    q = urllib.parse.quote(dish_name + " recipe")
    return f"https://www.google.com/search?q={q}"

def youtube_link(dish_name):
    q = urllib.parse.quote(f"how to make {dish_name} recipe")
    return f"https://www.youtube.com/results?search_query={q}"

def extract_dish_from_reply(text):
    """
    Extracts a clean dish name from ChefBot replies.
    """
    for line in text.split("\n"):
        line = line.strip()

        # Match common patterns
        if line.lower().startswith("you can make"):
            dish = line.replace("You can make", "").strip()

        elif line.lower().startswith("you can prepare"):
            dish = line.replace("You can prepare", "").strip()

        else:
            continue

        # Stop at first punctuation
        for stop in [".", " for ", " using ", " with "]:
            if stop in dish.lower():
                dish = dish.split(stop, 1)[0]

        return dish.strip()

    return None


# ===========================
# Load API Key (from Streamlit secrets)
# ===========================
API_KEY = st.secrets["api_keys"]["api_key"]
OPENROUTER_API_KEY = st.secrets["api_keys"]["openrouter_key"]
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_IMAGE = "https://via.placeholder.com/200x140.png?text=No+Image"

# ===========================
# Load Preprocessed Dataset
# ===========================
@st.cache_data
def load_recipes():
    with open("recipes_25k.pkl", "rb") as f:
        recipes = pd.read_pickle(f)
    return recipes

recipes_df = load_recipes()

# ===========================
# Ingredient Synonyms
# ===========================
ingredient_synonyms = {
    "bhindi": "okra", "ghee": "clarified butter", "idli rice": "parboiled rice",
    "paneer": "cottage cheese", "panner": "cottage cheese", "malai": "cream",
    "curd": "yogurt", "dahi": "yogurt", "jaggery": "unrefined cane sugar",
    "imli": "tamarind", "til": "sesame", "rajma": "kidney beans",
    "lobia": "black eyed peas", "chole": "chickpeas", "soya chunks": "textured vegetable protein",
    "rava": "semolina", "suji": "semolina", "atta": "wheat flour",
    "maida": "all-purpose flour", "besan": "gram flour", "sabudana": "tapioca pearls",
    "poha": "flattened rice", "haldi": "turmeric", "adrak": "ginger",
    "lasun": "garlic", "mirchi": "chili", "drumstick": "moringa",
    "tofu": "soy paneer", "jeera": "cumin", "methi": "fenugreek",
    "pulao": "fried rice", "kachumber": "salad", "dal": "lentils", "roti": "flatbread"
}

def normalize_ingredient(ing: str) -> str:
    ing = ing.lower().strip()
    return ingredient_synonyms.get(ing, ing)

# ===========================
# Taste Memory Functions
# ===========================
TASTE_FILE = "taste_memory.json"

def load_user_taste():
    if os.path.exists(TASTE_FILE):
        with open(TASTE_FILE, "r") as f:
            return json.load(f)
    return {"liked": []}

def save_user_taste(data):
    with open(TASTE_FILE, "w") as f:
        json.dump(data, f, indent=4)

if "user_taste" not in st.session_state:
    st.session_state.user_taste = load_user_taste()

# ===========================
# Spoonacular API (Primary)
# ===========================
def get_recipes_from_api(ingredients, diet="None"):
    try:
        url = "https://api.spoonacular.com/recipes/complexSearch"
        params = {
            "apiKey": API_KEY,
            "includeIngredients": ingredients,
            "number": 10,
            "addRecipeInformation": True
        }
        if diet != "None":
            params["diet"] = diet.lower()
        r = requests.get(url, params=params)
        if r.status_code != 200:
            return []
        data = r.json()
        if "results" not in data or not data["results"]:
            return []
        recipes = []
        for x in data["results"]:
            recipes.append({
                "title": x.get("title"),
                "readyInMinutes": x.get("readyInMinutes", "N/A"),
                "servings": x.get("servings", "N/A"),
                "sourceUrl": x.get("sourceUrl", "#"),
                "image": x.get("image", DEFAULT_IMAGE),
                "cuisines": ", ".join(x.get("cuisines", [])) or "Not specified",
                "nutrition": "N/A",
                "ingredients": [
                    f"{ing['amount']} {ing['unit']} {ing['name']}"
                    for ing in x.get("extendedIngredients", [])
            ],

            })
        return recipes
    except Exception as e:
        st.error(f"API Error: {e}")
        return []

# ===========================
# Google Search Fallback
# ===========================
def google_search_link(recipe_name):
    """Generate a Google search link for recipe instructions."""
    query = recipe_name.replace(" ", "+") + "+recipe"
    return f"https://www.google.com/search?q={query}"

# ===========================
# Enrich Offline Recipe (Optional)
# ===========================
def enrich_offline_recipe(recipe_name):
    try:
        url = "https://api.spoonacular.com/recipes/complexSearch"
        params = {
            "apiKey": API_KEY,
            "query": recipe_name,
            "addRecipeInformation": True,
            "number": 1
        }
        r = requests.get(url, params=params)
        if r.status_code != 200:
            return None
        data = r.json()
        if not data.get("results"):
            return None
        x = data["results"][0]
        return {
            "title": x.get("title", recipe_name),
            "readyInMinutes": x.get("readyInMinutes", "N/A"),
            "servings": x.get("servings", "N/A"),
            "cuisines": ", ".join(x.get("cuisines", [])) or "Not specified",
            "nutrition": "N/A",
            "image": x.get("image", DEFAULT_IMAGE),
            "sourceUrl": x.get("sourceUrl", "#"),
            "ingredients": []
        }
    except Exception as e:
        print(f"Offline enrichment error: {e}")
        return None

# ===========================
# ChefBot (OpenRouter)
# ===========================
def chefbot_response(recipe, user_question=None):
    system_prompt = f"""
You are a professional cooking assistant called ChefBot.

Recipe Name: {recipe['title']}
Cuisine: {recipe.get('cuisines', 'Not specified')}
Original Servings: {recipe.get('servings', 'N/A')}
Cooking Time: {recipe.get('readyInMinutes', 'N/A')} minutes
Ingredients with quantities:
{', '.join(recipe.get('ingredients', []))}

Tasks:
- Give cooking tips
- Suggest improvements
- Explain preparation
- Answer user doubts
- Suggest substitutions

Quantity Formatting Rules:
- Use metric units only (g, kg, ml, l).
- Avoid decimal numbers.
- Convert decimals to whole numbers or common fractions (½, ¼, ¾).
- Round quantities sensibly for home cooking.
- Always give a clean, readable ingredient list.

Behavior Rules:
- If the user asks for servings for N people, scale ingredients accordingly.
- Do NOT ask for ingredient lists.
- Output ingredients with quantities clearly.
- Be concise and clear in your responses.

ANSWERING RULES FOR COOKING TIME QUESTIONS:
- If the user asks about cooking time, give a DIRECT answer first.
- State total cooking time clearly.
- Mention whether cooking time changes or stays the same.
- Do NOT give long explanations.
- Do NOT ask follow-up questions.
- Keep the answer under 4 lines.
- If the user asks about ingredient substitutions, provide practical alternatives.
- If the user asks for variations, suggest realistic options based on the cuisine.
- If the user asks for health tips, suggest simple modifications to make the dish healthier.
- If the user asks for presentation tips, provide easy-to-follow suggestions.
- Always maintain a friendly and helpful tone.

LINK RULES:
- If the user asks about cooking steps, preparation, or how to make a dish:
  - Provide ONE official recipe website link.
  - Provide ONE YouTube video link.
- Use Google-style search links.
- Label links clearly as:
  "Official Recipe:" and "YouTube Video:"
- Do NOT ask the user to search themselves.
- Do NOT provide multiple links.
- Do NOT provide links for other types of questions.
- Always ensure links are relevant to the recipe provided.

"""

    messages = [{"role": "system", "content": system_prompt}]

    if user_question:
        messages.append({"role": "user", "content": user_question})
    else:
        messages.append({
            "role": "user",
            "content": "Give useful cooking suggestions for this dish."
        })

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": messages
    }

    response = requests.post(OPENROUTER_URL, headers=headers, json=payload)

    if response.status_code != 200:
        return "ChefBot is currently unavailable."

    return response.json()["choices"][0]["message"]["content"]


# ===========================
# ChefBot Recommendation Engine
# ===========================
def chefbot_recommend(recipes, user_taste):
    recipe_names = []

    # Handle API vs offline results
    for r in recipes[:5]:
        if isinstance(r, dict):  # API recipe
            recipe_names.append(r["title"])
        else:  # Offline dataframe row
            recipe_names.append(r)

    liked_titles = [r["title"] for r in user_taste.get("liked", [])[:5]]

    prompt = f"""
You are a food recommendation engine.

User liked recipes:
{liked_titles}

Available candidate recipes:
{recipe_names}

Task:
- Recommend EXACTLY 3 recipes
- Use this format strictly:

1. Recipe Name - short reason
2. Recipe Name - short reason
3. Recipe Name - short reason

Rules:
- Recipe names MUST match exactly from candidate recipes
- Do NOT invent new recipes
- Do NOT add extra text

"""

    return chefbot_response(
        {"title": "Recipe Recommendation"},
        prompt
    )

# ===========================
# Offline Smart Search
# ===========================
def search_offline_recipes(user_input, user_taste):
    ingredients = [normalize_ingredient(i) for i in user_input.split(",")]
    results = []
    liked_ings = [normalize_ingredient(i) for recipe in user_taste["liked"] for i in recipe.get("ingredients", [])]

    for idx, row in recipes_df.iterrows():
        recipe_ings = [normalize_ingredient(i) for i in row["ingredients"]]
        exact = sum(1 for ing in ingredients if ing in recipe_ings)
        partial = sum(1 for ing in ingredients if any(fuzz.partial_ratio(ing, r_ing) > 75 for r_ing in recipe_ings))
        taste_bonus = sum(1 for ing in recipe_ings if ing in liked_ings)
        score = exact * 2 + partial + taste_bonus * 0.5
        if score > 0:
            results.append((idx, score))

    if not results:
        return pd.DataFrame()
    top = sorted(results, key=lambda x: x[1], reverse=True)[:20]
    df = recipes_df.loc[[i for i, _ in top]].copy()
    df["Match Score"] = [s for _, s in top]
    return df[["title", "ingredients", "Match Score"]]
def extract_recipe_titles(text):
    titles = []
    for line in text.split("\n"):
        if "." in line and "-" in line:
            # "1. Lemon Pilaf Chicken - reason"
            title = line.split(".", 1)[1].split("-", 1)[0].strip()
            titles.append(title)
    return titles


# ===========================
# UI Styling
# ===========================
st.set_page_config(page_title="Smart Recipe Finder", page_icon="🍴", layout="wide")
st.markdown("""
<style>
.stApp {
    background-image: url("https://4kwallpapers.com/images/wallpapers/ios-13-stock-ipados-dark-green-black-background-amoled-ipad-2560x1440-794.jpg");
    background-size: cover;
    background-attachment: fixed;
}
.recipe-card {
    background-color: rgba(255,255,255,0.1);
    padding: 1em;
    border-radius: 12px;
    margin-bottom: 1em;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.4);
    text-align: center;
}
.recipe-img {
    border-radius: 10px;
    width: 200px;
    height: 140px;
    object-fit: cover;
    margin-bottom: 8px;
}
.recipe-title {
    font-size: 1.1em;
    font-weight: 600;
    color: #00ffcc;
}
.recipe-meta {
    color: #ddd;
    font-size: 0.9em;
}
</style>
""", unsafe_allow_html=True)

st.title("🍴 Smart Recipe Finder")
tab1, tab2, tab3 = st.tabs(["🔍 Find Recipes", "❤️ Liked Recipes", "👨‍🍳 ChefBot"])


# -------------------- TAB 1 --------------------
with tab1:
    ingredients = st.text_input("Enter ingredients (comma-separated):")
    diet = st.selectbox("Dietary preference:", ["None", "Vegetarian", "Vegan", "Gluten-Free", "Ketogenic"])

    if "search_results" not in st.session_state:
        st.session_state.search_results = None
        st.session_state.search_mode = None

    if st.button("Find Recipes"):
        if ingredients:
            with st.spinner("Searching recipes..."):
                api_results = get_recipes_from_api(ingredients, diet)
                if api_results:
                    st.session_state.search_results = api_results
                    st.session_state.search_mode = "api"
                else:
                    offline = search_offline_recipes(ingredients, st.session_state.user_taste)
                    st.session_state.search_results = offline
                    st.session_state.search_mode = "offline"
        else:
            st.warning("Please enter some ingredients.")

    # Display Search Results
    if st.session_state.search_results is not None:
        if st.session_state.search_mode == "api":
            st.subheader("🌐 Recipes from Spoonacular API")
            cols = st.columns(2)

            # ------------------ RECIPE CARDS ------------------
            for i, r in enumerate(st.session_state.search_results):
                with cols[i % 2]:
                    st.markdown(f"""
                    <div class='recipe-card'>
                        <img src='{r['image']}' class='recipe-img'>
                        <div class='recipe-title'>{r['title']}</div>
                        <div class='recipe-meta'>⏱️ {r['readyInMinutes']} mins | 🍽️ Serves {r['servings']}</div>
                        <div class='recipe-meta'>Cuisine: {r['cuisines']}</div>
                        <a href='{r['sourceUrl']}' target='_blank'>🔗 View Full Recipe</a>
                    </div>
                    """, unsafe_allow_html=True)

                    if st.button(f"❤️ Like {r['title']}", key=f"api_{r['title']}"):
                        if not any(x["title"] == r["title"] for x in st.session_state.user_taste["liked"]):
                            st.session_state.user_taste["liked"].append(r)
                            save_user_taste(st.session_state.user_taste)

                    if st.button(f"👨‍🍳 Ask ChefBot", key=f"chef_{r['title']}"):
                        st.session_state.selected_recipe = r

            # ------------------ ADD THIS PART ------------------
            st.markdown("---")

            if st.button("🤖 Recommend Best Recipes"):
                with st.spinner("ChefBot is analyzing your taste..."):
                    recommendation_text = chefbot_recommend(
                        st.session_state.search_results,
                        st.session_state.user_taste
                    )

                # Extract titles and store them
                st.session_state.recommended_titles = extract_recipe_titles(recommendation_text)

                st.success("ChefBot recommendations updated 👇")

        else:
            offline = st.session_state.search_results
            st.subheader("📁 Recipes from Offline Dataset")
            cols = st.columns(2)

            for i, (_, row) in enumerate(offline.iterrows()):
                img = f"https://source.unsplash.com/200x140/?food,{row['title'].replace(' ', '%20')}"
                g_link = google_search_link(row['title'])

                with cols[i % 2]:
                    st.markdown(f"""
                    <div class='recipe-card'>
                        <img src='{img}' class='recipe-img'>
                        <div class='recipe-title'>{row['title']}</div>
                        <div class='recipe-meta'>Ingredients: {', '.join(row['ingredients'][:6])}...</div>
                        <div class='recipe-meta'>Match Score: {round(row['Match Score'],2)}</div>
                        <a href='{g_link}' target='_blank'>🔗 Google Recipe Instructions</a>
                    </div>
                    """, unsafe_allow_html=True)

                    # ❤️ Like Button (CORRECT PLACE)
                    if st.button(f"❤️ Like {row['title']}", key=f"off_{row['title']}"):
                        if not any(x["title"] == row["title"] for x in st.session_state.user_taste["liked"]):
                            enriched = enrich_offline_recipe(row["title"])
                            if enriched:
                                st.session_state.user_taste["liked"].append(enriched)
                            else:
                                st.session_state.user_taste["liked"].append({
                                    "title": row["title"],
                                    "ingredients": row["ingredients"],
                                    "readyInMinutes": "N/A",
                                    "servings": "N/A",
                                    "cuisines": "Not specified",
                                    "nutrition": "N/A",
                                    "image": img,
                                    "sourceUrl": g_link
                                })
                            save_user_taste(st.session_state.user_taste)
                            st.success(f"Saved {row['title']} to liked recipes!")

                    # 👨‍🍳 ChefBot Button (CORRECT PLACE)
                    if st.button(f"👨‍🍳 Ask ChefBot", key=f"chef_off_{row['title']}"):
                        st.session_state.selected_recipe = {
                            "title": row["title"],
                            "ingredients": row["ingredients"],
                            "cuisines": "Not specified",
                            "servings": "N/A",
                            "readyInMinutes": "N/A"
                        }
    # =======================
    # 🍽️ RECOMMENDED RECIPES
    # =======================
    if "recommended_titles" in st.session_state and st.session_state.recommended_titles:

        if not st.session_state.search_results:
            st.warning("Search recipes first to get recommendations.")
        else:
            st.markdown("---")
            st.subheader("🍽️ ChefBot Recommended Recipes")

            cols = st.columns(2)
            shown = 0

            for r in st.session_state.search_results:
                if r["title"].strip().lower() in [
                    t.strip().lower() for t in st.session_state.recommended_titles
                ]:
                    with cols[shown % 2]:
                        st.markdown(f"""
                        <div class='recipe-card'>
                            <img src='{r.get("image", DEFAULT_IMAGE)}' class='recipe-img'>
                            <div class='recipe-title'>{r["title"]}</div>
                            <div class='recipe-meta'>⏱️ {r.get("readyInMinutes","N/A")} mins | 🍽️ Serves {r.get("servings","N/A")}</div>
                            <div class='recipe-meta'>Cuisine: {r.get("cuisines","Not specified")}</div>
                            <a href='{r.get("sourceUrl","#")}' target='_blank'>🔗 View Full Recipe</a>
                        </div>
                        """, unsafe_allow_html=True)

                        if st.button(f"❤️ Like {r['title']}", key=f"rec_like_{r['title']}"):
                            if not any(x["title"] == r["title"] for x in st.session_state.user_taste["liked"]):
                                st.session_state.user_taste["liked"].append(r)
                                save_user_taste(st.session_state.user_taste)

                        if st.button(f"👨‍🍳 Ask ChefBot", key=f"rec_chef_{r['title']}"):
                            st.session_state.selected_recipe = r

                    shown += 1


# -------------------- TAB 2 --------------------
with tab2:
    st.subheader("💾 Your Liked Recipes")
    liked = st.session_state.user_taste["liked"]

    if not liked:
        st.info("You haven’t liked any recipes yet.")
    else:
        cols = st.columns(2)
        for i, r in enumerate(liked):
            img = r.get("image", DEFAULT_IMAGE)
            ready = r.get("readyInMinutes", "N/A")
            serve = r.get("servings", "N/A")
            cuisine = r.get("cuisines", "Not specified")
            nutrition = r.get("nutrition", "N/A")
            source = r.get("sourceUrl", "#")

            with cols[i % 2]:
                st.markdown(f"""
                <div class='recipe-card'>
                    <img src='{img}' class='recipe-img'>
                    <div class='recipe-title'>{r['title']}</div>
                    <div class='recipe-meta'>⏱️ {ready} mins | 🍽️ Serves {serve}</div>
                    <div class='recipe-meta'>Cuisine: {cuisine}</div>
                    <div class='recipe-meta'>Nutrition: {nutrition}</div>
                    <a href='{source}' target='_blank'>🔗 View Full Recipe</a>
                </div>
                """, unsafe_allow_html=True)

                if st.button(f"💔 Remove {r['title']}", key=f"rem_{r['title']}"):
                    st.session_state.user_taste["liked"] = [
                        x for x in st.session_state.user_taste["liked"] if x["title"] != r["title"]
                    ]
                    save_user_taste(st.session_state.user_taste)
                    st.rerun()
# -------------------- TAB 3 --------------------
with tab3:
    st.subheader("👨‍🍳 ChefBot – Your Cooking Assistant")

    if "selected_recipe" not in st.session_state:
        st.info("Select a recipe from the Find Recipes tab to talk with ChefBot.")
    else:
        recipe = st.session_state.selected_recipe

        st.markdown(f"### 🍽️ {recipe['title']}")
        st.write("Ask anything about this dish — ingredients, steps, tips, or variations.")

        user_question = st.text_input(
            "Ask ChefBot:",
            placeholder="e.g. How can I improve the taste? Can I replace any ingredient?"
        )

        if user_question:
            if user_question:
                with st.spinner("ChefBot is thinking..."):
                    reply = chefbot_response(recipe, user_question)

                st.markdown("#### 👨‍🍳 ChefBot says:")
                st.success(reply)

                # 🔍 Detect dish name from reply (ingredient-based questions)
                detected_dish = extract_dish_from_reply(reply)

                # Decide which dish name to use for links
                link_dish = detected_dish if detected_dish else recipe["title"]

                st.markdown("### 🔗 Helpful Links")
                st.markdown(f"- 📖 **Official Recipe:** {recipe_link(link_dish)}")
                st.markdown(f"- ▶️ **YouTube Video:** {youtube_link(link_dish)}")

