import os
import traceback

from amadeusgpt import app_utils, st
from amadeusgpt.utils import validate_openai_api_key

# Set page configuration
st.set_page_config(layout="wide")
app_utils.load_css()


def main():
    if "log_folder" not in st.session_state:
        st.session_state["log_folder"] = "logs"
    if "user_input" not in st.session_state:
        st.session_state["user_input"] = ""
    if "example" not in st.session_state:
        st.session_state["example"] = ""
    if "exist_valid_openai_api_key" not in st.session_state:
        if "OPENAI_API_KEY" in os.environ:
            st.session_state["exist_valid_openai_api_key"] = True
            st.session_state["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]
        elif "OPENROUTER_API_KEY" in os.environ:
            st.session_state["exist_valid_openai_api_key"] = True
            st.session_state["OPENROUTER_API_KEY"] = os.environ["OPENROUTER_API_KEY"]
        else:
            st.session_state["exist_valid_openai_api_key"] = False

    example_to_page = {}

    def valid_api_key():
        print("inside valid api key function")
        if "OPENAI_API_KEY" in os.environ:
            api_token = os.environ["OPENAI_API_KEY"]
        elif "OPENROUTER_API_KEY" in os.environ:
            api_token = os.environ["OPENROUTER_API_KEY"]
        else:
            api_token = st.session_state["openAI_token"]
        check_valid = validate_openai_api_key(api_token)

        if check_valid:
            st.session_state["exist_valid_openai_api_key"] = True
            st.session_state["OPENAI_API_KEY"] = api_token
            st.success("OpenAI API Key Validated!")
        else:
            st.error("Invalid OpenAI API Key")

    def welcome_page(text):
        with st.sidebar as sb:
            if st.session_state["exist_valid_openai_api_key"] is not True:
                api_token = st.sidebar.text_input(
                    "Your openAI API token",
                    "place your token here",
                    key="openAI_token",
                    on_change=valid_api_key,
                )
        current_script_directory = os.path.dirname(os.path.abspath(__file__))
        logo_path = os.path.join(
            current_script_directory, "static/images/amadeusgpt_logo.png"
        )
        st.image(
            logo_path,
            caption=None,
            width=None,
            use_column_width=None,
            clamp=False,
            channels="RGB",
            output_format="auto",
        )

        st.markdown(
            "##### 🪄 We turn natural language descriptions of behaviors into machine-executable code"
        )

        small_head = "#" * 6
        small_font = ""

        st.markdown("### 👥 Instructions")

        st.markdown(
            f"{small_font} - We use LLMs to bridge natural language and behavior analysis code. For more details, check out our NeurIPS 2023 paper '[AmadeusGPT: a natural language interface for interactive animal behavioral analysis' by Shaokai Ye, Jessy Lauer, Mu Zhou, Alexander Mathis \& Mackenzie W. Mathis](https://github.com/AdaptiveMotorControlLab/AmadeusGPT)."
        )
        st.markdown(
            f"{small_font} - 🤗 Please note that depending on openAI, the runtimes can vary - you can see the app is `🏃RUNNING` in the top right when you run demos or ask new queries.\n"
        )
        st.markdown(
            f"{small_font} - Please give us feedback if the output is correct 👍, or needs improvement 👎. This is an ` academic research project` demo, so expect a few bumps please, but we are actively working to make it better 💕.\n"
        )
        st.markdown(
            f"{small_font} - ⬅️ To get started, watch the quick video below ⬇️, and then select a demo from the drop-down menu. 🔮 We recommend to refresh the browser/app between demos."
        )

        st.markdown(
            f"{small_font} - To create an OpenAI API key please see: https://platform.openai.com/overview.\n"
        )

        st.markdown("### How AmadeusGPT🎻 works")

        st.markdown(
            f"{small_font} - To capture animal-environment states, AmadeusGPT🎻 leverages state-of-the-art pretrained models, such as SuperAnimals for animal pose estimation and Segment-Anything (SAM) for object segmentation. The platform enables spatio-temporal reasoning to parse the outputs of computer vision models into quantitative behavior analysis. Additionally, AmadeusGPT🎻 simplifies the integration of arbitrary behavioral modules, making it easier to combine tools for task-specific models and interface with machine code."
        )
        st.markdown(
            f"{small_font} - We built core modules that interface with several integrations, plus built a dual-memory system to augment chatGPT, thereby allowing longer reasoning."
        )
        st.markdown(
            f"{small_font} - This demo serves to highlight a hosted user-experience, but does not include all the features yet..."
        )
        # st.markdown(f"{small_font} - Watch the video below to see how to use the App.")

        # st.video("static/demo_withvoice.mp4")

        st.markdown("### ⚠️ Disclaimers")

        st.markdown(
            f"{small_font} Refer to https://streamlit.io/privacy-policy for the privacy policy for your personal information.\n"
            f"{small_font} Please note that to improve AmadeusGPT🎻 we log your queries and the generated code on our demos."
            f"{small_font} Note, we do *not* log your openAI API key under any circumstances and we rely on streamlit cloud for privately securing your connections.\n"
            f"{small_font} If you have security concerns over the API key, we suggest that you re-set your API key after you finish using our app.\n"
        )

        st.markdown("### 💻 The underlying core computer vision models explained")
        st.markdown(
            f"{small_font} We use pretrained computer vision models to capture the state of the animal and the environment. We hope this can reduce the entry barrier to behavior analysis.\n"
            f"{small_font} Therefore, we can ask questions about animals' behaviors that are composed by animal's state, animal-animal interactions or animal-environment interactions.\n"
        )
        st.markdown(
            f"{small_font} DeepLabCut-SuperAnimal models, see https://arxiv.org/abs/2203.07436"
        )
        st.markdown(
            f"{small_font} MetaAI Segment-Anything models, see https://arxiv.org/abs/2304.02643"
        )

        st.markdown("### FAQ")
        st.markdown(f"{small_font} Q: What can be done by AmadeusGPT🎻?")
        st.markdown(
            f"{small_font} - A: We provide a natural language interface to analyze video-based behavioral data. \n"
            f"{small_font} We expect the user to describe a behavior before asking about the behaviors.\n"
            f"{small_font} In general, one can define behaviors related to the movement of an animal (see EPM example), animal to animal interactions (see the MABe example) and\n"
            f"{small_font} animal-environment interaction (check MausHaus example)."
        )
        st.markdown(f"{small_font} Q: Can I run my own videos?")
        st.markdown(
            f"{small_font} - A: Not yet - due to limited compute resources we disabled on-demand pose estimation and object segmentation thus we cannot take new videos at this time. For your best experience, we pre-compute the pose and segmentation for example videos we provided. However, running DeepLabCut, SAM and other computer vision models is possible with AmadeusGPT🎻 so stay tuned!"
        )
        st.markdown(
            f"{small_font} Q: in the demos you use the term 'unit' - What is the unit being used?"
        )
        st.markdown(
            f"{small_font} - A: Pixels for distance and pixel per frame for speed and velocity given we don't have real-world values in distance"
        )
        st.markdown(
            f"{small_font} Q: How can I draw ROI and use the ROI to define a behavior?"
        )
        st.markdown(f"{small_font} - A: Check the video on the EPM tab!")
        st.markdown(f"{small_font} Q: How can I ask AmadeusGPT🎻 to plot something?")
        st.markdown(
            f"{small_font} - A: Check the demo video and prompts in the examples"
        )
        st.markdown(
            f"{small_font} Q: Why did AmadeusGPT🎻 produce errors or give me unexpected answers to my questions?"
        )
        st.markdown(
            f"{small_font} - A: Most likely that you are asking for something that is beyond the current capability of AmadeusGPT🎻 or\n"
            "you are asking questions in a way that is unexpected. In either cases, we appreciate it if you can provide feedback \n"
            "to us in our GitHub repo so we can improve our system (and note we log your queries and will use this to improve AmadeusGPT🎻)."
        )

        st.markdown(f"{small_font} Q: Does it work with mice only?")
        st.markdown(
            f"{small_font} - A: No, AmadeusGPT🎻 can work with a range of animals as long as poses are extracted and behaviors can be defined with those poses. We will add examples of other animals in the future."
        )
        st.markdown(
            f"{small_font} Q: How do I know I can trust AmadeusGPT🎻's answers?"
        )
        st.markdown(
            f"{small_font} - A: For people who are comfortable with reading Python code, reading the code can help validate the answer. We welcome the community to check our APIs. Otherwise, try visualize your questions by asking \n"
            f"{small_font} AmadeusGPT🎻 to plot the related data and use the visualization as a cross validation. We are also developing new features\n"
            f"{small_font} to help you gain more confidence on the results and how the results are obtained."
        )
        st.markdown(
            f"{small_font} Q: Why the page is blocked for a long time and there is no response?"
        )
        st.markdown(
            f"{small_font} - A: There might be a high traffic for either ChatGPT API or the Streamlit server. Refresh the page and retry or come back later."
        )

    if st.session_state["exist_valid_openai_api_key"]:
        example_list = ["Welcome", "EPM", "MausHaus", "MABe", "Horse"]
    else:
        example_list = ["Welcome"]

    for key in example_list:
        if key == "Welcome":
            example_to_page[key] = welcome_page
        else:
            example_to_page[key] = app_utils.render_page_by_example

    with st.sidebar as sb:
        example_bar = st.sidebar.selectbox(
            "Select an example dataset", example_to_page.keys()
        )

    try:
        if "enable_profiler" in os.environ:
            example_to_page[example_bar](example_bar)
        else:
            example_to_page[example_bar](example_bar)

    except Exception as e:
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
