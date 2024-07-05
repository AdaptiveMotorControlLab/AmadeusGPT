import base64
import gc
import io
import json
import os
import tempfile
from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

import amadeusgpt
from amadeusgpt.analysis_objects.analysis_factory import create_analysis
from amadeusgpt.analysis_objects.object import Object, ROIObject
from amadeusgpt.config import Config
from amadeusgpt.main import create_amadeus

LOG_DIR = os.path.join(os.path.expanduser("~"), "Amadeus_logs")
VIDEO_EXTS = "mp4", "avi", "mov"
current_script_directory = os.path.dirname(os.path.abspath(__file__))
user_profile_path = os.path.join(
    current_script_directory, "static", "images", "cat.png"
)
bot_profile_path = os.path.join(
    current_script_directory, "static", "images", "chatbot.png"
)


def load_profile_image(image_path):
    if image_path.startswith("http"):
        response = requests.get(image_path)
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(image_path)
    return img


def load_css():
    current_script_directory = os.path.dirname(os.path.abspath(__file__))
    css_path = os.path.join(current_script_directory, "static/styles/style.css")
    if os.path.exists(css_path):
        st.markdown(f"<style>{open(css_path).read()}</style>", unsafe_allow_html=True)
    else:
        st.error(f"File not found: {css_path}")


USER_PROFILE = load_profile_image(user_profile_path)
BOT_PROFILE = load_profile_image(bot_profile_path)


class BaseMessage:
    def __init__(self, json_entry=None):
        if json_entry:
            self.data = self.parse_from_json_entry(json_entry)
        else:
            self.data = {}

    def __getitem__(self, key):
        return self.data[key]

    def parse_from_json_entry(self, json_entry):
        return json_entry

    def __str__(self):
        return str(self.data)

    def format_caption(self, caption):
        temp = caption.split("\n")
        temp = [f"<li>{content}</li>" for content in temp]
        ret = "<ul>\n" + "".join(temp).rstrip() + "\n</ul>"
        return ret

    def render(self):
        raise NotImplementedError("Must implement this")


class HumanMessage(BaseMessage):
    def __init__(self, query=None, json_entry=None):
        if json_entry:
            super().__init__(json_entry=json_entry)
        else:
            self.data = {}
        self.data["role"] = "human"
        if query:
            self.data["query"] = query

    def render(self):
        if len(self.data) > 0:
            for render_key, render_value in self.data.items():
                if render_key == "query":
                    st.markdown(
                        f'<div class="panel">{render_value}</div>',
                        unsafe_allow_html=True,
                    )


class AIMessage(BaseMessage):
    def __init__(self, amadeus_answer=None, json_entry=None):
        self.rendered = False
        if json_entry:
            super().__init__(json_entry=json_entry)
        else:
            self.data = {}
        self.data["role"] = "ai"
        if not isinstance(amadeus_answer, dict):
            amadeus_answer = amadeus_answer.to_dict()
        self.data.update(amadeus_answer)

    def render(self, debug = False):
        """
        We use the getter for better encapsulation
        overall structure of what to be rendered

              | chain of thoughts
        error | str_answer | nd_array
              | helper_code
              | code
              | figure
              |      | figure_explanation
              | figure
              |      | figure_explanation
              | overall_explanation

        --------
        """

        render_keys = [
            "query",
            "chain_of_thought",
            "error_message",
            "function_rets",
            "plots",
            "out_videos",
        ]
        # for render_key, render_value in self.data.items():
        if len(self.data) > 0:
            for render_key in render_keys:
                if render_key not in self.data:
                    continue
                render_value = self.data[render_key]
                if render_value is None:
                    # skip empty field
                    continue
                if render_key == "function_rets":
                    if isinstance(render_key, (list, tuple)):
                        for ret in render_value:
                            for key, value in ret.items():
                                st.code(f"Result\n {value}\n ", language="python")
                    else:
                        st.code(f"Result\n {render_value}\n ", language="python")
                elif render_key == "error_message":
                    st.markdown(f"The error says: {render_value}\n ")
                elif render_key == "chain_of_thought":
                    # there should be a better matching than this
                    text = render_value
                    lines = text.split("\n")
                    inside_code_block = False
                    code_block = []
                    for line in lines:
                        if line.strip().startswith("```python"):
                            inside_code_block = True
                            code_block = []
                        elif line.strip().startswith("```") and inside_code_block:
                            inside_code_block = False
                            st.code("\n".join(code_block), language="python")
                        elif inside_code_block:
                            code_block.append(line)
                        else:
                            st.markdown(line)
                    sandbox = self.data["sandbox"]
                    qa_message = sandbox.code_execution(self.data)
                    if qa_message["error_message"] is not None:
                        # make the error message more visible with different color
                        st.markdown(f"Error: {qa_message['error_message']}\n ")
                        # Remind users we are fixing the error by self debugging
                        st.markdown(f"Let me try to fix the error by self-debugging\n ")
                        if not debug:
                            sandbox.llms["self_debug"].speak(sandbox)                        
                            qa_message = sandbox.code_execution(qa_message)
                            self.render(debug = True)
                    # do not need to execute the block one more time
                    if not self.rendered:
                        self.rendered = True
                        sandbox.render_qa_message(qa_message)

                elif render_key == "ndarray":
                    for content_array in render_value:
                        content_array = content_array.squeeze()
                        # no point of showing array that's large

                        hint_message = "Here is the output:"
                        st.markdown(
                            f'<div class="panel">{hint_message}</div>',
                            unsafe_allow_html=True,
                        )
                        if isinstance(content_array, str):
                            st.markdown(
                                f'<div class="panel">{content_array}</div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            if len(content_array.shape) == 2:
                                df = pd.DataFrame(content_array)
                                st.dataframe(df, use_container_width=True)
                            else:
                                raise ValueError(
                                    "returned array cannot be non 2D array."
                                )
                elif render_key == "plots":
                    # are there better ways in streamlit now to support plot display?
                    for fig, axe in render_value:
                        filename = save_figure_to_tempfile(fig)
                        st.image(filename, width=600)
                elif render_key == "out_videos":
                    for video_path in render_value:
                        if os.path.exists(video_path):
                            st.video(video_path)


class Messages:
    """
    The data class that is used in the front end (i.e., streamlit)
    methods
    -------
    parse_from_csv
        parse from example.csv for demo prompts
    render_message:
        render messages
    to_csv:
        to csv file

    attributes
    ----------
    {'plot': List[str],
    a list of file paths to image files that can be used by streamlit to render plots

    'code':  List[str],
    a list of str that contains functions. Though in the future,
    there might be more functions than the task program functions
    maybe it should be dictionary

    'ndarray': List[nd.array],
    Some sort of rendering for ndarray might be useful

    'text': List[str]
    The text needs to be further decomposed into different
    kinds of text, such as the main text, the plot documentation, the error
    each might have different colors and different positions
    }

    parameters
    ----------
    data: a dictionary
    """

    def __init__(self):
        self.raw_dict = None
        self.messages = []

    def parse_from_json(self, path):
        with open(path, "r") as f:
            json_obj = json.load(f)
        for json_entry in json_obj:
            if "query" in json_entry:
                self.append(HumanMessage(json_entry=json_entry))
            else:
                self.append(AIMessage(json_entry=json_entry))

    def render(self):
        """
        make sure those media types match what is in amadeus answer class
        """
        for message in self.messages:
            message.render()

    def append(self, e):
        self.messages.append(e)

    def insert(self, ind, e):
        self.messages.insert(ind, e)

    def __len__(self):
        return len(self.messages)

    def __iter__(self):
        return iter(self.messages)

    def __getitem__(self, ind):
        return self.messages[ind]

    def __setitem__(self, ind, value):
        self.messages[ind] = value


def get_amadeus_instance(example):
    # construct the config from the current example
    # get the root directory of the the module amadeusgpt
    config = get_config(example)
    amadeus_instance = create_amadeus(config)
    return amadeus_instance


def ask_amadeus(question):
    amadeus = get_amadeus_instance(st.session_state["example"])
    qa_message = amadeus.step(question)

    return qa_message


# caching display roi will make the roi stick to
# the display of initial state
def display_roi(analysis, example):
    analysis = create_analysis(get_config(example))
    roi_objects = analysis.get_roi_objects()

    frame = analysis.visual_manager.get_scene_image()
    colormap = plt.cm.get_cmap("rainbow", len(roi_objects))

    for i, roi_object in enumerate(roi_objects):
        name = roi_object.get_name()
        vertices = roi_object.Path.vertices
        pts = np.array(vertices, np.int32)
        pts = pts.reshape((-1, 1, 2))
        color = colormap(i)[:3]
        color = tuple(int(c * 255) for c in color[::-1])
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=5)
        text = name
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_position = (pts[0, 0, 0], pts[0, 0, 1])
        cv2.putText(frame, text, text_position, font, 1, color, 2, cv2.LINE_AA)
    with st.sidebar:
        st.caption("ROIs in the scene")
        st.image(frame)


def update_roi(analysis, result_json, ratios):
    w_ratio, h_ratio = ratios
    objects = pd.json_normalize(result_json["objects"])
    for col in objects.select_dtypes(include=["object"]).columns:
        objects[col] = objects[col].astype("str")
    roi_objects = {}
    objects = objects.to_dict(orient="dict")
    if "path" in objects:
        paths = objects["path"]
        count = 0
        for path_id in paths:
            temp = eval(paths[path_id])
            paths[path_id] = temp
            canvas_path = paths[path_id]
            if not isinstance(canvas_path, list):
                continue
            points = [[p[1], p[2]] for p in canvas_path if len(p) == 3]
            points = np.array(points)
            points[:, 0] = points[:, 0] * w_ratio
            points[:, 1] = points[:, 1] * h_ratio
            _object = ROIObject(f"ROI{count}", canvas_path=points)
            count += 1
            analysis.object_manager.add_roi_object(_object)

    analysis.object_manager.save_roi_objects("temp_roi_objects.pickle")


def place_st_canvas(analysis, key, scene_image):

    width, height = scene_image.size
    # we always resize the canvas to its default values and keep the ratio

    w_ratio = width / 600
    h_ratio = height / 400

    with st.sidebar:
        st.caption(
            "Left click to draw a polygon. Right click to confirm the drawing. Refresh the page if you need new ROIs or if the ROI canvas does not display"
        )
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.9)",
            stroke_width=3,
            background_image=scene_image,
            # update_streamlit = realtime_update,
            width=600,
            height=400,
            drawing_mode="polygon",
            key=f"{key}_canvas",
        )

    if (
        canvas_result.json_data is not None
        and "path" in canvas_result.json_data
        and len(canvas_result.json_data["path"]) > 0
    ):
        pass
    if canvas_result.json_data is not None:
        update_roi(analysis, canvas_result.json_data, (w_ratio, h_ratio))

    if len(analysis.object_manager.get_roi_object_names()) > 0:
        display_roi(analysis, key)


def chat_box_submit():
    if "user_input" in st.session_state:
        query = st.session_state["user_input"]
        qa_message = ask_amadeus(query)

        user_message = HumanMessage(query=qa_message["query"])
        amadeus_message = AIMessage(amadeus_answer=qa_message)

        st.session_state["messages"].append(user_message)
        st.session_state["messages"].append(amadeus_message)


def rerun_prompt(query, ind):
    messages = st.session_state["messages"]
    amadeus_answer = ask_amadeus(query)
    amadeus_message = AIMessage(amadeus_answer=amadeus_answer)

    if ind != len(messages) - 1 and messages[ind + 1]["role"] == "ai":
        messages[ind + 1] = amadeus_message
    else:
        messages.insert(ind + 1, amadeus_message)


def render_messages():
    example = st.session_state["example"]
    messages = st.session_state["messages"]

    if len(messages) == 0 and example != "Custom":
        example_history_json = os.path.join(f"examples/{example}/example.json")
        messages.parse_from_json(example_history_json)

    for ind, msg in enumerate(messages):
        _role = msg["role"]
        with st.chat_message(
            _role, avatar=USER_PROFILE if _role == "human" else BOT_PROFILE
        ):
            # parse and render message which is a list of dictionary
            if _role == "human":
                msg.render()
                disabled = not st.session_state["exist_valid_openai_api_key"]
                button_name = "Generate Response"
                st.button(
                    button_name,
                    key=f"{example}_user_{ind}",
                    on_click=rerun_prompt,
                    kwargs={"query": msg["query"], "ind": ind},
                    disabled=disabled,
                )
            else:
                msg.render()

    st.session_state["messages"] = messages

    disabled = not st.session_state["exist_valid_openai_api_key"]
    st.chat_input(
        "Ask me new questions here ...",
        key="user_input",
        on_submit=chat_box_submit,
        disabled=disabled,
    )
    csv = None
    return csv


def update_df_data(new_item, index_to_update, df, csv_file):
    flat_items = [item[0] for item in new_item]
    df.iloc[index_to_update] = None
    df.loc[index_to_update, "Index"] = index_to_update
    for item in flat_items:
        key = item["type"]
        value = item["content"]
        if key in df.columns:
            df.loc[index_to_update, key] = value
    df.to_csv(csv_file, index=False)

    return csv_file, df


def get_scene_image(config):
    analysis = create_analysis(config)

    scene_image = analysis.visual_manager.get_scene_image()
    if scene_image is not None:
        scene_image = Image.fromarray(scene_image)
        buffered = io.BytesIO()
        scene_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str


def get_config(example):

    root_dir = os.path.dirname(amadeusgpt.__file__)
    if example == "Horse":
        template_config_path = os.path.join(root_dir, "configs", "Horse_template.yaml")
    elif example == "MausHaus":
        template_config_path = os.path.join(
            root_dir, "configs", "maushaus_template.yaml"
        )
    elif example == "EPM":
        template_config_path = os.path.join(root_dir, "configs", "EPM_template.yaml")
    elif example == "MABe":
        template_config_path = os.path.join(root_dir, "configs", "mabe_template.yaml")
    elif example == "Custom":
        template_config_path = os.path.join(root_dir, "configs", "Custom_template.yaml")

    config = Config(template_config_path)

    return config


def save_uploaded_file(uploaded_file, save_dir):

    filename = uploaded_file.name
    save_path = os.path.join(save_dir, filename)
    if not os.path.exists(save_path):
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    return save_path


def render_page_by_example(example):
    # get the config
    if example not in st.session_state:
        st.session_state[example] = {}
    # needs different logic for custom page
    config = get_config(example)

    current_script_directory = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(
        current_script_directory, "static", "images", "amadeusgpt_logo.png"
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
    # st.markdown("# Welcome to AmadeusGPT🎻")

    if example == "Custom":
        st.markdown("Provide your own video and keypoint file (in pairs)")
        save_dir = os.path.join("examples", example)
        if "uploaded_keypoint_file" not in st.session_state:
            uploaded_keypoint_file = st.file_uploader(
                "Choose keypoint files",
                ["h5"],
                accept_multiple_files=False,
            )
            if uploaded_keypoint_file is not None:
                path = save_uploaded_file(uploaded_keypoint_file, save_dir)
                st.session_state["uploaded_keypoint_file"] = path
                config["keypoint_info"]["keypoint_file_path"] = st.session_state["uploaded_keypoint_file"]

        if "uploaded_video_file" not in st.session_state:
            uploaded_video_file = st.file_uploader(
                "Choose video files",
                VIDEO_EXTS,
                accept_multiple_files=False,
            )
            if uploaded_video_file is not None:
                path = save_uploaded_file(uploaded_video_file, save_dir)
                st.session_state["uploaded_video_file"] = uploaded_video_file
                config["video_info"]["video_file_path"] = st.session_state["uploaded_video_file"]      

        ###### USER INPUT PANEL ######
        # get user input once getting the uploaded files
        if uploaded_keypoint_file is None or uploaded_video_file is None:
            disabled = True
        else:
            disabled = False

        if disabled:
            st.warning("Please upload a file before entering text.")

    elif example == "EPM":
        st.markdown(
            "Elevated plus maze (EPM) is a widely used behavioral test. The mouse is put on an elevated platform with two open arms (without walls) and  two closed arms (with walls). \
                    In this example we used a video from https://www.nature.com/articles/s41386-020-0776-y."
        )
        st.markdown(
            "- ⬅️ On the left you can see the video data auto-tracked with DeepLabCut and keypoint names (below). You can also draw ROIs to ask questions to AmadeusGPT🎻 about the ROIs. You can drag the divider between the panels to increase the video/image size."
        )
        st.markdown(
            "- We suggest you start by clicking 'Generate Response' to our demo queries."
        )
        st.markdown(
            "- Ask additional questions in the chatbox at the bottom of the page."
        )
        st.markdown(
            "- Here are some example queries you might consider: 'The <|open arm|> is the ROI0. How much time does the mouse spend in the open arm?' (NOTE here you can re-draw an ROI0 if you want. Be sure to click 'finish drawing') | 'Define head_dips as a behavior where the mouse's mouse_center and neck are in ROI0 which is open arm while head_midpoint is outside ROI1 which is the cross-shape area. When does head_dips happen and what is the number of bouts for head_dips?' "
        )
        st.markdown("- ⬇️🎥 Watch this short clip on how to draw the ROI(s)🤗")

        st.video(
            os.path.join(current_script_directory, "static/customEPMprompt_short.mp4")
        )

    elif example == "MABe":
        st.markdown(
            "MABe Mouse Triplets is part of a behavior benchmark presented in Sun et al 2022 https://arxiv.org/abs/2207.10553. In the videos, three mice exhibit multiple social behaviors including chasing.   \
                    In this example, we take one video where chasing happens between mice."
        )
        st.markdown(
            "- ⬅️ On the left you can see the video data and keypoint names, which could be useful for your queries. You can drag the divider between the panels to increase the video/image size."
        )
        st.markdown(
            "- We suggest you start by clicking 'Generate Response' to our demo queries."
        )
        st.markdown(
            "- Ask additional questions in the chatbox at the bottom of the page."
        )
        analysis = create_analysis(config)

    elif example == "MausHaus":
        st.markdown(
            "MausHaus is a dataset that records a freely moving mouse within a rich environment with objects. More details can be found https://arxiv.org/pdf/2203.07436.pdf."
        )
        st.markdown(
            "- ⬅️ On the left you can see the video data auto-tracked with DeepLabCut, segmented image with SAM, and the keypoint guide, which could be useful for your queries. You can drag the divider between the panels to increase the video/image size."
        )
        st.markdown(
            "- We suggest you start by clicking 'Generate Response' to our demo queries."
        )
        st.markdown(
            "- Ask additional questions in the chatbox at the bottom of the page."
        )
        st.markdown(
            "- Here are some example queries you might consider: 'Give me events where the animal overlaps with the treadmill, which is object 5' | 'Define <|drinking|> as a behavior where the animal's nose is over object 28, which is a waterbasin. The minimum time window for this behavior should be 20 frames. When is the animal drinking?'"
        )
        config = get_config(example)

    elif example == "Horse":
        st.markdown(
            "This horse video is part of a benchmark by Mathis et al 2021 https://arxiv.org/abs/1909.11229."
        )

    config = get_config(example)
    analysis = create_analysis(config)

    if st.session_state["example"] != example:
        st.session_state["messages"] = Messages()
    st.session_state["example"] = example

    scene_image_path = get_scene_image(config)

    if scene_image_path is not None:
        img_data = base64.b64decode(scene_image_path)
        image_stream = io.BytesIO(img_data)
        image_stream.seek(0)
        scene_image = Image.open(image_stream)
    else:
        scene_image = None

    col1, col2, col3 = st.columns([2, 1, 1])

    # if example == "MausHaus" or st.session_state['enable_SAM'] == "Yes":
    #     if 'sam_info' in config and os.path.exists(config['sam_info'].get('sam_checkpoint', '')):
    #         sam_image = get_sam_image(config)
    #     else:
    #         st.error("Cannot find SAM checkpoints. Skipping SAM")

    with st.sidebar as sb:
        if example == "MABe":
            st.caption("Raw video from MABe")
        elif example == "Horse":
            st.caption("Raw video from Horse-30")
        else:
            st.caption("DeepLabCut-SuperAnimal tracked video")
        if config["video_info"]["video_file_path"] and config["video_info"]["video_file_path"] is not None: 
            st.video(config["video_info"]["video_file_path"])

        if "uploaded_video_file" in st.session_state:
            st.video(st.session_state["uploaded_video_file"])
        
        # we only show objects for MausHaus for demo
        # if sam_image is not None:
        #     st.caption("SAM segmentation results")
        #     st.image(sam_image, channels="RGBA")

    if (
        st.session_state["example"] == "EPM"
        or st.session_state["example"] == "MausHaus"
        and scene_image is not None
    ):
        place_st_canvas(analysis, example, scene_image)

    if st.session_state["example"] == "Custom" and scene_image:
        place_st_canvas(analysis, example, scene_image)

    if example == "EPM" or example == "MausHaus":
        # will read the keypoints from h5 file to avoid hard coding
        with st.sidebar:
            topviewimage = os.path.join(
                current_script_directory, "static", "images", "supertopview.png"
            )
            st.image(topviewimage)
            # st.image("static/images/supertopview.png")
    with st.sidebar:
        st.write("Keypoints:")
        st.write(analysis.get_keypoint_names())

    render_messages()
    gc.collect()


def save_figure_to_tempfile(fig):
    # save the figure
    folder_path = os.path.join("logs", "tmp_imgs")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # Generate a unique temporary filename in the specified folder
    temp_file = tempfile.NamedTemporaryFile(
        dir=folder_path, suffix=".png", delete=False
    )
    filename = temp_file.name
    temp_file.close()
    fig.savefig(
        filename,
        format="png",
        bbox_inches="tight",
        pad_inches=0.0,
        dpi=400,
        transparent=True,
    )
    return filename


def make_plot_pretty4dark_mode(fig, ax):
    fig = plt.gcf()
    fig.set_facecolor("none")
    ax = plt.gca()
    ax.set_facecolor("none")
    # Set axes and legend colors to white or other light colors
    ax.spines["bottom"].set_color("white")
    ax.spines["top"].set_color("white")
    ax.spines["right"].set_color("white")
    ax.spines["left"].set_color("white")

    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")
    legend = plt.legend()
    for text in legend.get_texts():
        text.set_color("white")

    return fig, ax


def display_image(temp_file):
    full_image = Image.open(temp_file)
    st.image(full_image)


def display_temp_text(text_content):
    # Convert the text content to base64
    text_bytes = text_content.encode("utf-8")
    text_base64 = base64.b64encode(text_bytes).decode()
    # Display the link to the text file
    st.markdown(
        f'<a href="data:text/plain;charset=utf-8;base64,{text_base64}" target="_blank">Check error.</a>',
        unsafe_allow_html=True,
    )


def style_button_row(clicked_button_ix, n_buttons):
    def get_button_indices(button_ix):
        return {"nth_child": button_ix, "nth_last_child": n_buttons - button_ix + 1}

    clicked_style = """
    div[data-testid*="stHorizontalBlock"] > div:nth-child(%(nth_child)s):nth-last-child(%(nth_last_child)s) button {
        border-color: rgb(255, 75, 75);
        color: rgb(255, 75, 75);
        box-shadow: rgba(255, 75, 75, 0.5) 0px 0px 0px 0.2rem;
        outline: currentcolor none medium;
    }
    """
    unclicked_style = """
    div[data-testid*="stHorizontalBlock"] > div:nth-child(%(nth_child)s):nth-last-child(%(nth_last_child)s) button {
        pointer-events: none;
        cursor: not-allowed;
        opacity: 0.65;
        filter: alpha(opacity=65);
        -webkit-box-shadow: none;
        box-shadow: none;
    }
    """
    style = ""
    for ix in range(n_buttons):
        ix += 1
        if ix == clicked_button_ix:
            style += clicked_style % get_button_indices(ix)
        else:
            style += unclicked_style % get_button_indices(ix)
    st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)
