from openai import OpenAI
import streamlit as st
import requests


st.set_page_config(page_title="Fay's Bible", page_icon="☻", layout="centered")

SYSTEM_PROMPT = {
    "role": "system",
    "content": """You are a scholarly educator on the Bible.

Rules:
- Do not provide spiritual guidance
- Provide context, clarification and insight into bible verses
- Always cite specific Bible verses (Book Chapter:Verse) when relevant
- Provide historical and cultural context when helpful
- Be respectful of all Christian denominations as well as other religions
- Keep responses clear and accessible
- If unsure, say so rather than making things up

Tone: Warm, thoughtful, and encouraging."""
}

st.markdown("""
<style>
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"],
    .stChatMessage img,
    .stChatMessage svg {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

if "verse_results" not in st.session_state:
    st.session_state.verse_results = None
 
st.title("`☻ welcome`")
st.markdown("`to your personal bible - for supplementing your studies.`")
st.markdown("---")
st.markdown("""<style>h1 { color: #1866cc }</style> <h1>lookup a chapter or verse:</h1>""", unsafe_allow_html=True)
# want this color: #1866cc


with st.sidebar:
    st.header("Search Instructions")
    st.markdown("""
    - Search an `entire chapter` like :red[**Philippians 4**]
    
    - Search a `single verse` like :red[**Jeremiah 29:11**]
    
    - Search for a `range of verses` like :red[**Matthew 6:25-34**]
    
    - Search for `multiple chapters` like :red[**John 3:16-4:10**]
    
    """)
    st.markdown("---")
    st.markdown("""
                <div style='text-align: center; color: gray;'>
                <small>Made with ❤️ by Fay</small>
                </div>
                """, unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns([1, 0.5, 0.5, 0.5])

with col1:
    TRANSLATIONS = {
        "kjv": "King James Version",
        "web": "World English Bible",
        "bbe": "Bible in Basic English",
        "asv": "American Standard Version", 
    }
    translation = st.selectbox(
        "Select Translation",
        options=TRANSLATIONS.keys(),
        format_func=lambda x: TRANSLATIONS[x]
    )

with col2:
    book = st.text_input("Book Name", placeholder="Genesis")

with col3:
    verse = st.text_input("Chapter + Verse", placeholder="1:1")

with col4:
    st.markdown("<br>", unsafe_allow_html=True)
    search_button = st.button("Search", type="secondary")


def get_verse(book, verse, translation):
    url = f'https://bible-api.com/{book}+{verse}?translation={translation}'
    try:
        response = requests.get(url)
        if response.status_code == 404:
            st.error("Error! Please enter a valid book and verse.")
            return None
        elif response.status_code == 200: # if successful
            return response.json()
        else:
            # catch errors
            st.warning(f"Unexpected error. (Status code: {response.status_code})")
            return None
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

def display_verse(bible_content):
    if bible_content:
        st.markdown("---")
        st.badge(f"{bible_content['reference']}", color="blue")
        for v in bible_content["verses"]:
            with st.container():
                st.write(f'`{v["verse"]}` {v["text"]}')


if search_button:
    if book and verse:
        with st.spinner("..."):
            result = get_verse(book, verse, translation)
            if result:
                st.session_state.verse_results = result
    elif book and not verse:
        st.warning("Please enter a chapter and verse.")
    else:
        st.warning("Please enter both a book name and verse.")

display_verse(st.session_state.verse_results)
        
st.markdown("---")


# implement large language model

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = [SYSTEM_PROMPT]

for message in st.session_state.messages:
    if message["role"] != "system":  
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("need context or clarification?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        # providing current verse or chapter for context if available
        messages_to_send = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]
        if st.session_state.verse_results:
            verse_text = "\n".join(
                f'{v["verse"]}. {v["text"]}' for v in st.session_state.verse_results["verses"]
            )
            verse_context = {
                "role": "system",
                "content": f"The user is currently viewing {st.session_state.verse_results['reference']}:\n{verse_text}"
            }
            messages_to_send.insert(1, verse_context)

        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=messages_to_send,
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
    



        
        
        
        
    
