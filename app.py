# NOTES (not for today cuz LATE)
# actually add a section talking about why this app was made
# and like who is Jesus and stuff like that
# or at least provide a link to it 
# but yeah what's the goal again YE preach the Word amen.


from openai import OpenAI
import streamlit as st
import requests
from datetime import datetime
from zoneinfo import ZoneInfo
# scratched the clipboard thing but might come back to it
# from st_copy_to_clipboard import st_copy_to_clipboard
from supabase import create_client
from streamlit_js_eval import streamlit_js_eval

# setup supabase
supabase = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

# setup page
st.set_page_config(page_title="daily bread 𓍯𓂃✍︎", page_icon="bread.ico", layout="centered")

# login authentication featurss
def init_auth_state():
    """Initialize authentication session state"""
    if "user" not in st.session_state:
        st.session_state.user = None
    if "auth_mode" not in st.session_state:
        st.session_state.auth_mode = "login"  # 'login', 'signup', or 'reset'

@st.dialog(" ")
def auth_modal():
    """Combined modal for login, signup, and password reset"""
    #logo
    # st.image("bread.ico", width=67)
    # NOTE: need to fix the forgot password, actually add the logic
    tab1, tab2 = st.tabs(["Log in", "I'm new here"]) 

    with tab1:
        # LOGIN
        login_email = st.text_input("Email", key="login_email")
        login_password = st.text_input("Password", type="password", key="login_password")

        if st.button("Sign In", key="login_btn"):
            if login_email and login_password:
                try:
                    # sign in part
                    response = supabase.auth.sign_in_with_password({
                        "email": login_email,
                        "password": login_password
                    })

                    # store user info and session inside session state
                    st.session_state.user = response.user
                    st.session_state.access_token = response.session.access_token
                    st.session_state.refresh_token = response.session.refresh_token

                    # set the session (in supa base client)
                    supabase.postgrest.auth(response.session.access_token)
                    st.rerun()

                except Exception as e:
                    st.error(f"Login failed: {str(e)}")
            else:
                st.error("Please enter both email and password")

    with tab2:
        # SIGN UP
        st.write("Don't have an account? Create one today ☻")
        signup_email = st.text_input("Email", key="signup_email")
        signup_password = st.text_input("Password (min 6 characters)", type="password", key="signup_password")
        signup_password_confirm = st.text_input("Confirm Password", type="password", key="signup_password_confirm")

        if st.button("Create account", key="signup_btn"):
            if signup_email and signup_password and signup_password_confirm:
                if signup_password != signup_password_confirm:
                    st.error("Passwords don't match!")
                elif len(signup_password) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    try:
                        # Supabase sign up
                        response = supabase.auth.sign_up({
                            "email": signup_email,
                            "password": signup_password
                        })

                        if response.user:
                            st.session_state.user = response.user
                            if response.session:
                                st.session_state.access_token = response.session.access_token
                                st.session_state.refresh_token = response.session.refresh_token
                                supabase.postgrest.auth(response.session.access_token)
                            st.rerun()
                        else:
                            st.error("Sign up failed. Please try again.")

                    except Exception as e:
                        st.error(f"Sign up failed: {str(e)}")
            else:
                st.error("Please fill in all fields")

    # FORGOT PASSWORD 
    # with tab3:
    #     # PASSWORD RESET
    #     st.write("Enter your email to receive a password reset link")
    #     reset_email = st.text_input("Email", key="reset_email")
    #
    #     if st.button("Send Reset Link", key="reset_btn"):
    #         if reset_email:
    #             try:
    #                 supabase.auth.reset_password_email(reset_email)
    #                 st.success("Password reset link sent! Check your email.")
    #             except Exception as e:
    #                 st.error(f"Failed to send reset link: {str(e)}")
    #         else:
    #             st.error("Please enter your email")

def logout():
    """Log out the current user"""
    try:
        supabase.auth.sign_out()
        st.session_state.user = None
        st.session_state.access_token = None
        st.session_state.refresh_token = None
        st.rerun()
    except Exception as e:
        st.error(f"Logout failed: {str(e)}")

def save_verse_reference(reference, translation, notes=""):
    """Save a verse reference to the database"""
    if not st.session_state.user:
        st.error("Please sign in to save verses")
        return False

    try:
        # avoid duplicates
        existing = supabase.table("saved_verses").select("id").eq("user_id", st.session_state.user.id).eq("reference", reference).execute()
        if existing.data:
            st.warning(f"{reference} is already saved!", icon="⚠️")
            return False

        supabase.table("saved_verses").insert({
            "user_id": st.session_state.user.id,
            "reference": reference,
            "verse_text": "", 
            "translation": translation,
            "notes": notes
        }).execute()
        return True
    except Exception as e:
        st.error(f"Failed to save verse: {str(e)}")
        return False


def get_saved_verses():
    """Get all saved verses for the current user"""
    if not st.session_state.user:
        return []

    try:
        response = supabase.table("saved_verses").select("*").eq("user_id", st.session_state.user.id).order("created_at", desc=True).execute()
        return response.data
    except Exception as e:
        st.error(f"Failed to load saved verses: {str(e)}")
        return []

def delete_saved_verse(verse_id):
    """Delete a saved verse"""
    try:
        supabase.table("saved_verses").delete().eq("id", verse_id).execute()
        return True
    except Exception as e:
        st.error(f"Failed to delete verse: {str(e)}")
        return False

def parse_reference(reference):
    """Parse reference like 'Genesis 1:1' into book and verse"""
    parts = reference.split()
    if len(parts) >= 2:
        if parts[0].isdigit() and len(parts) >= 3:
            book = f"{parts[0]} {parts[1]}"
            verse = parts[2]
        else:
            book = parts[0]
            verse = parts[1]
        return book, verse
    return None, None

@st.dialog(" ")
def save_verse_modal(reference, translation):
    """Modal to save a verse with optional notes"""
    st.write(f"Saving **{reference}** ({translation})")

    notes_input = st.text_area("Add notes", key="modal_verse_notes", placeholder="", height=67)

    if st.button("Save", key="confirm_save_btn", use_container_width=True):
        if save_verse_reference(reference, translation, notes_input):
            st.rerun()

@st.dialog(" ")
def verse_detail_modal(verse):
    st.subheader(f"{verse['reference']} ({verse['translation']})")
    
    # get notes
    if verse.get('notes') and verse['notes'].strip():
        st.markdown("**your notes:**")
        st.info(verse['notes'])
    else:
        st.caption("No notes for this bookmark.")

    col1, col2 = st.columns(2)
    
    # actions in the modal
    with col1:
        if st.button("Load", key=f"load_detail_{verse['id']}", use_container_width=True):
            book, verse_ref = parse_reference(verse['reference'])
            if book and verse_ref:
                result = get_verse(book, verse_ref, verse['translation'])
                if result:
                    st.session_state.verse_results = result
                    st.session_state.current_translation = verse['translation']
                    st.rerun()

    with col2:
        if st.button("Delete", key=f"delete_detail_{verse['id']}", type="secondary", use_container_width=True):
            if delete_saved_verse(verse['id']):
                st.rerun()


# setup gpt wrapper
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

# css custom styling changes
st.markdown("""
<style>
    /*hide the AI profile pictures too corny LOL*/
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"],
    .stChatMessage img,
    .stChatMessage svg {
        display: none !important;
    }

    /*hide the text inside the feedback form (too crowded)*/
    .stForm [data-testid="stFormSubmitButton"]::after {
        display: none !important;
    }
    .stForm small {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# init
init_auth_state()
if "access_token" in st.session_state and st.session_state.access_token:
    supabase.postgrest.auth(st.session_state.access_token)

if "verse_results" not in st.session_state:
    st.session_state.verse_results = None
if "user_tz" not in st.session_state:
    st.session_state.user_tz = None
tz_string = streamlit_js_eval(js_expressions="Intl.DateTimeFormat().resolvedOptions().timeZone", key="tz")
if tz_string:
    st.session_state.user_tz = tz_string
 
 
# title and header of page
st.write("**welcome!** open sidebar for more.")
st.markdown("""<style>h1 { color: #1866cc }</style> <h1>lookup a chapter or verse:</h1>""", unsafe_allow_html=True)
# note: #1866cc


# sidebar!
with st.sidebar:
    
    # full date and time
    try:
        user_tz = ZoneInfo(st.session_state.user_tz) if st.session_state.user_tz else ZoneInfo("America/Los_Angeles")
    except:
        user_tz = ZoneInfo("America/Los_Angeles")
    now = datetime.now(user_tz)
    st.markdown(f":blue[helping you get your **daily bread**, even at {now.strftime('%I:%M%p').lstrip('0').lower()} on a random {now.strftime('%A').lower()}.]")


    # check if user is logged in , display dif things
    if st.session_state.user: # logged in
        if st.button("Log Out", key="logout_btn"):
            logout()
        st.markdown("---")
        
        # show saved verses/bookmarks
        st.subheader("Bookmarks")
        saved_verses = get_saved_verses()

        if not saved_verses:
            st.caption("No bookmarks here yet!")
        else:
            for verse in saved_verses:
                if st.button(f"{verse['reference']}", key=f"verse_{verse['id']}"):
                    verse_detail_modal(verse)

    else: # if user is not logged in, don't show
        if st.button("Log in", key="open_auth_modal"):
            auth_modal()

    st.markdown("---")
    st.header("Search Instructions")
    st.markdown("""
    - Search an `entire chapter` like :red[**Philippians 4**]
    
    - Search a `single verse` like :red[**Jeremiah 29:11**]
    
    - Search for a `range of verses` like :red[**Matthew 6:25-34**]
    
    - Search for `multiple chapters` like :red[**John 3:16-4:10**]
    
    """)
    
    # mini feedback bar for funsies
    st.markdown("---")
    with st.form(key="feedback_form", clear_on_submit=True):
        name = st.text_input(label="Send me feedback!", placeholder="Your name")
        message = st.text_input(label="", placeholder="Your message here", label_visibility="collapsed")
        submitted = st.form_submit_button("Send")

        if submitted and name and message:
            supabase.table("feedback").insert({"name":name, "message":message}).execute()
            st.success(f"Thanks for the feedback, {name}!")
            
    # footer
    st.markdown("---")
    st.markdown("""
                <div style='text-align: center; color: gray;'>
                <small>Made with ❤️ by Fay</small>
                </div>
                """, unsafe_allow_html=True)
    
    

# front page columns (search tool)
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


# display stuff
with col2:
    book = st.text_input("Book Name", placeholder="Genesis")

with col3:
    verse = st.text_input("Chapter + Verse", placeholder="1:1")

with col4:
    st.markdown("<br>", unsafe_allow_html=True)
    search_button = st.button("Search", type="secondary")


# get the verse with bible api
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

# logic to show the bible verse
def display_verse(bible_content, translation="kjv"):
    if bible_content:
        st.markdown("---")
        st.badge(f"{bible_content['reference']}", color="blue")
        reference = bible_content['reference']
        base_ref = reference.split(':')[0] if ':' in reference else reference
        enduring_word_path = ""

        # check if the split is 3 or 2
        str_split = base_ref.split(" ") # split into list of 2 or 3 strings
        cur_len = len(str_split)
        ch = ""
        ve = ""
        pr = ""
        if cur_len == 2:
            ch=str_split[0]
            ve=str_split[1]
        elif cur_len == 3:
            pr = f"{str_split[0]}-"
            ch=str_split[1]
            ve=str_split[2]


        # note bug: in enduring word they call it "psalm" without  the s.
        if ch == "Psalms":
            ch = "Psalm"

        enduring_word_path = f'https://enduringword.com/bible-commentary/{pr}{ch}-{ve}/'
        for v in bible_content["verses"]:
            st.write(f'`{v["verse"]}` {v["text"]}')
            
        st.markdown("---")
        
        
        # show save verse button only if logged in
        if st.session_state.user:
            if st.button("Make bookmark"):
                save_verse_modal(reference, translation.upper())
                
        # link to enduring word bible commentary
        st.page_link(label=f'Read commentary on this chapter from :blue[**Enduring Word**]', page=enduring_word_path)
            
        st.markdown("---")
        
        
# trigger with the search btn
if search_button:
    if book and verse:
        with st.spinner("..."):
            result = get_verse(book, verse, translation)
            if result:
                st.session_state.verse_results = result
                st.session_state.current_translation = translation
    elif book and not verse:
        st.warning("Please enter a chapter and verse.")
    else:
        st.warning("Please enter both a book name and verse.")


current_translation = st.session_state.get("current_translation", "kjv")
display_verse(st.session_state.verse_results, current_translation)


# implement large language model kekw
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = [SYSTEM_PROMPT]

for message in st.session_state.messages:
    if message["role"] != "system":  
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("ask an ai.."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        # making sure to provide current verse or chapter for context if available
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
    



        
        
        
        
    
