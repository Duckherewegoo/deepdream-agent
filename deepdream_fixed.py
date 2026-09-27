import os
import json
import uuid
from typing import List, Any, Dict

# 第三方依赖
from openai import OpenAI
import gradio as gr
from PIL import Image, ImageFilter
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ====================== LCEL 核心导入（LangChain 1.2.0 最新标准） ======================
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import (
    BaseMessage, AIMessage, HumanMessage, ToolMessage
)
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# ====================== 全局配置 ======================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# 阿里云百炼客户端
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY", "sk-your-api-key-here"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 5种梦境配置
DREAM_CONFIGS = {
    "normal": {"MAX_SIZE": 800, "OCTAVE_LAYERS": 2, "OCTAVE_SCALE": 0.85, "ITER_PER_OCTAVE": 40, "STEP_SIZE": 0.008, "REALITY_FUSION": 0.15, "SAFE_SHIFT": False},
    "sweet": {"MAX_SIZE": 800, "OCTAVE_LAYERS": 2, "OCTAVE_SCALE": 0.9, "ITER_PER_OCTAVE": 35, "STEP_SIZE": 0.006, "REALITY_FUSION": 0.2, "SAFE_SHIFT": False},
    "nightmare": {"MAX_SIZE": 800, "OCTAVE_LAYERS": 2, "OCTAVE_SCALE": 0.8, "ITER_PER_OCTAVE": 60, "STEP_SIZE": 0.012, "REALITY_FUSION": 0.1, "SAFE_SHIFT": True},
    "mist": {"MAX_SIZE": 1000, "OCTAVE_LAYERS": 3, "OCTAVE_SCALE": 0.75, "ITER_PER_OCTAVE": 80, "STEP_SIZE": 0.012, "REALITY_FUSION": 0.1, "SAFE_SHIFT": True},
    "crazy": {"MAX_SIZE": 1000, "OCTAVE_LAYERS": 6, "OCTAVE_SCALE": 0.75, "ITER_PER_OCTAVE": 200, "STEP_SIZE": 0.03, "REALITY_FUSION": 0.0, "SAFE_SHIFT": True},
}

# ====================== DeepDream 核心生成函数 ======================
def load_image(image_path: str, cfg: Dict) -> tf.Tensor:
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img, dtype=np.float32)
    h, w = img_np.shape[:2]
    max_side = max(h, w)
    if max_side > cfg["MAX_SIZE"]:
        scale = cfg["MAX_SIZE"] / max_side
        img_np = tf.image.resize(img_np, (int(h*scale), int(w*scale))).numpy()
    return tf.convert_to_tensor(tf.keras.applications.inception_v3.preprocess_input(img_np))

def build_dream_model(dream_type: str):
    base = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet")
    if dream_type == "normal":
        return tf.keras.Model(base.input, [base.get_layer("mixed0").output, base.get_layer("mixed1").output])
    if dream_type == "sweet":
        return tf.keras.Model(base.input, [base.get_layer("mixed0").output])
    if dream_type == "nightmare":
        return tf.keras.Model(base.input, [base.get_layer("mixed3").output, base.get_layer("mixed4").output])
    if dream_type == "mist":
        return tf.keras.Model(base.input, [base.get_layer("mixed2").output, base.get_layer("mixed3").output, base.get_layer("mixed4").output])
    if dream_type == "crazy":
        return tf.keras.Model(base.input, [base.get_layer(f"mixed{i}").output for i in range(7)])
    return tf.keras.Model(base.input, [base.get_layer("mixed0").output])

@tf.function
def dream_step(image, model, step_size, reality_fusion, origin):
    with tf.GradientTape() as tape:
        tape.watch(image)
        loss = tf.reduce_sum([tf.reduce_mean(act) for act in model(tf.expand_dims(image,0))])
    grads = tape.gradient(loss, image)
    grads /= tf.math.reduce_std(grads) + 1e-8
    image += grads * step_size
    return tf.clip_by_value(image*(1-reality_fusion)+origin*reality_fusion, -1, 1)

def generate_dream_base(image, model, cfg):
    original_shape = tf.shape(image)[:2]
    shapes = [tf.cast(tf.cast(original_shape, tf.float32)*(cfg["OCTAVE_SCALE"]**i), tf.int32) for i in range(cfg["OCTAVE_LAYERS"])][::-1]
    current_img = tf.image.resize(image, shapes[0])
    for size in shapes:
        current_img = tf.image.resize(current_img, size)
        current_origin = tf.image.resize(image, size)
        for _ in range(cfg["ITER_PER_OCTAVE"]):
            current_img = dream_step(current_img, model, cfg["STEP_SIZE"], cfg["REALITY_FUSION"], current_origin)
    return tf.image.resize(current_img, original_shape)

def generate_mist_dream(img, model, cfg):
    original_shape = tf.shape(img)[:2]
    shapes = [tf.cast(tf.cast(original_shape, tf.float32)*(cfg["OCTAVE_SCALE"]**i), tf.int32) for i in range(cfg["OCTAVE_LAYERS"])][::-1]
    current = tf.image.resize(img, shapes[0])
    for size in shapes:
        current = tf.image.resize(current, size)
        origin = tf.image.resize(img, size)
        for _ in range(cfg["ITER_PER_OCTAVE"]):
            if cfg["SAFE_SHIFT"]:
                current = tf.roll(tf.roll(current, np.random.randint(-2,2), 0), np.random.randint(-2,2), 1)
            current = dream_step(current, model, cfg["STEP_SIZE"], cfg["REALITY_FUSION"], origin)
    return tf.image.resize(current, original_shape)

def generate_crazy_dream(img, model, cfg):
    original_shape = tf.shape(img)[:2]
    shapes = [tf.cast(tf.cast(original_shape, tf.float32)*(cfg["OCTAVE_SCALE"]**i), tf.int32) for i in range(cfg["OCTAVE_LAYERS"])][::-1]
    current = tf.image.resize(img, shapes[0])
    for size in shapes:
        current = tf.image.resize(current, size)
        for _ in range(cfg["ITER_PER_OCTAVE"]):
            current = tf.roll(tf.roll(current, np.random.randint(-2,2), 0), np.random.randint(-2,2), 1)
            with tf.GradientTape() as tape:
                tape.watch(current)
                loss = tf.reduce_sum([tf.reduce_mean(act) for act in model(tf.expand_dims(current,0))])
            grads = tape.gradient(loss, current)
            grads /= tf.math.reduce_std(grads) + 1e-8
            current += grads * cfg["STEP_SIZE"]
            current = tf.clip_by_value(current, -1, 1)
    return tf.image.resize(current, original_shape)

def process_dream_result(tensor, dream_type: str) -> Image.Image:
    img = tensor.numpy()
    img = 255 * (img + 1.0) / 2.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img)
    if dream_type == "sweet":
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=1.2))
    if dream_type == "nightmare":
        pil_img = pil_img.filter(ImageFilter.SHARPEN)
    return pil_img

def generate_dream_by_type(input_path: str, dream_type: str) -> str:
    dream_type = dream_type if dream_type in DREAM_CONFIGS else "normal"
    cfg = DREAM_CONFIGS[dream_type]
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = f"./dream_outputs/{base_name}_{dream_type}.png"
    os.makedirs("./dream_outputs", exist_ok=True)
    try:
        img = load_image(input_path, cfg)
        model = build_dream_model(dream_type)
        if dream_type in ["normal","sweet","nightmare"]:
            res = generate_dream_base(img, model, cfg)
        elif dream_type == "mist":
            res = generate_mist_dream(img, model, cfg)
        else:
            res = generate_crazy_dream(img, model, cfg)
        process_dream_result(res, dream_type).save(output_path)
        return output_path
    except Exception:
        try:
            Image.open(input_path).convert("RGB").save(output_path)
        except Exception:
            pass
        return output_path

# ====================== LCEL 工具定义 ======================
@tool
def get_weather(city: str) -> str:
    """查询城市天气"""
    weather = {
        "北京":"晴 15-25°C，适合做梦的好天气！",
        "上海":"小雨 18-22°C，梦里听雨声~",
        "广州":"晴 25-32°C，热带梦境！",
        "深圳":"多云 24-30°C，科技感梦境！"
    }
    return json.dumps({
        "city": city,
        "info": weather.get(city, f"{city}：天气数据获取中，先做个美梦吧~")
    })

@tool
def generate_dream(dream_type: str, user_image_path: str) -> str:
    """生成梦境图片，类型：normal/sweet/nightmare/mist/crazy"""
    if not os.path.exists(user_image_path):
        return json.dumps({"success": False, "msg": "图片不存在"})
    output = generate_dream_by_type(user_image_path, dream_type)
    dream_cn = {"normal":"正常梦","sweet":"美梦","nightmare":"噩梦","mist":"迷雾梦","crazy":"疯狂梦"}
    return json.dumps({
        "success": True,
        "type": dream_cn.get(dream_type, dream_type),
        "path": output
    })

TOOLS = [generate_dream, get_weather]
TOOL_MAP = {t.name: t for t in TOOLS}

# ====================== 修复版：阿里云百炼 LCEL 兼容模型 ======================

class BaiLianChatModel(BaseChatModel):
    model_name: str = "qwen-turbo"
    temperature: float = 0.7

    # 实现工具绑定：把LangChain工具转为OpenAI兼容格式
    def _bind_tools(self, tools, **kwargs):
        openai_tools = [convert_to_openai_tool(tool) for tool in tools]
        return {"tools": openai_tools}

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: List[str] = [],
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        msg_list = []
        for m in messages:
            if m.type == "human":
                msg_list.append({"role":"user","content":m.content})
            elif m.type == "system":
                msg_list.append({"role":"system","content":m.content})
            elif m.type == "ai":
                msg_list.append({"role":"assistant","content":m.content})
            elif isinstance(m, ToolMessage):
                msg_list.append({
                    "role":"tool",
                    "content":m.content,
                    "tool_call_id": m.tool_call_id
                })
        
        # 调用百炼兼容接口，透传tools/stop等绑定参数
        resp = client.chat.completions.create(
            model=self.model_name,
            messages=msg_list,
            temperature=self.temperature,
            stop=stop if stop else None,
            **kwargs
        )

        choice_msg = resp.choices[0].message
        content = choice_msg.content or ""
        additional_kwargs = {}

        # 提取工具调用，写入additional_kwargs供后续执行
        if choice_msg.tool_calls:
            additional_kwargs["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in choice_msg.tool_calls
            ]

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=content,
                        additional_kwargs=additional_kwargs
                    )
                )
            ]
        )

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: List[str] = [],
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop, run_manager, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "bailian"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model": self.model_name}

# ====================== LCEL 核心链构建（官方标准） ======================
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是专业的梦境生成助手，可调用工具生成DeepDream图片、查询城市天气。用中文友好回复，工具结果格式化展示。"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# 初始化模型 + 绑定工具
llm = BaiLianChatModel()
llm_with_tools = llm.bind_tools(TOOLS)

# 工具执行函数（LCEL 核心）
def execute_tools(ai_msg: AIMessage) -> List[ToolMessage]:
    tool_msgs = []
    tool_calls = ai_msg.additional_kwargs.get("tool_calls", [])
    for tool_call in tool_calls:
        tool_name = tool_call["function"]["name"]
        tool = TOOL_MAP.get(tool_name)
        if not tool:
            continue
        args = json.loads(tool_call["function"]["arguments"])
        res = tool.invoke(args)
        tool_msgs.append(ToolMessage(content=res, tool_call_id=tool_call["id"]))
    return tool_msgs

# 构建 LCEL 完整调用链
lcel_chain = (
    RunnablePassthrough.assign(
        messages=lambda x: x["chat_history"] + [HumanMessage(content=x["input"])]
    )
    | RunnableLambda(lambda x: prompt.invoke(x))
    | llm_with_tools
)

# ====================== 工具调用 + 结果解析 ======================
def lcel_chat(input_text: str, image_path: str = None):
    if image_path:
        input_text = f"{input_text} | 图片路径：{image_path}"
    
    chat_history = []
    # 第一步：LLM 生成响应
    ai_msg = lcel_chain.invoke({"input": input_text, "chat_history": chat_history})
    
    # 第二步：如果调用工具，自动执行
    if ai_msg.additional_kwargs.get("tool_calls"):
        tool_msgs = execute_tools(ai_msg)
        chat_history.extend([ai_msg] + tool_msgs)
        # 第三步：LLM 根据工具结果生成最终回答
        ai_msg = lcel_chain.invoke({"input": input_text, "chat_history": chat_history})
    
    return ai_msg.content

# ====================== Gradio 界面 ======================
def save_image(image) -> str:
    if not image:
        return ""
    os.makedirs("./uploads", exist_ok=True)
    path = f"./uploads/upload_{uuid.uuid4().hex[:8]}.png"
    image.save(path)
    return path

def gui_chat(user_text, user_image):
    img_path = save_image(user_image)
    try:
        return lcel_chat(user_text, img_path)
    except Exception as e:
        return f"❌ 执行错误：{str(e)}"

# ====================== 启动程序 ======================
if __name__ == "__main__":
    # 创建必要文件夹
    for d in ["./uploads", "./dream_outputs"]:
        os.makedirs(d, exist_ok=True)
    
    # Gradio 界面
    with gr.Blocks(title="LCEL 纯架构 · DeepDream 梦境生成助手") as demo:
        gr.Markdown("""
        # 🌙 纯 LCEL 架构 · DeepDream 梦境生成助手
        **支持功能**：5种梦境生成 | 天气查询 | 阿里云百炼大模型
        """)
        with gr.Row():
            user_img = gr.Image(type="pil", label="上传图片（生成梦境必选）", height=300)
            with gr.Column():
                user_input = gr.Textbox(label="输入指令", placeholder="例：生成美梦 / 查询北京天气", lines=3)
                output = gr.Textbox(label="AI 响应", lines=10)
        with gr.Row():
            submit = gr.Button("🚀 生成", variant="primary")
            clear = gr.Button("🧹 清空")
        
        # 绑定事件
        submit.click(gui_chat, [user_input, user_img], output)
        clear.click(lambda: (None, "", ""), outputs=[user_img, user_input, output])
    
    # 启动服务
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
