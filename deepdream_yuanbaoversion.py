# ====================== 核心依赖 ======================
import os
import re
import time
import random
import json
import uuid
import base64
import io
from typing import Union, List, Any, Dict, Tuple
from dataclasses import dataclass
from openai import OpenAI
import gradio as gr
from PIL import Image, ImageFilter
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt

# 设置中文字体和避免警告
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# LangChain
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from langchain.embeddings.base import Embeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.tools import Tool
from langchain_core.callbacks import CallbackManagerForLLMRun

# Pydantic
from pydantic import BaseModel, Field, field_validator

# ====================== 百炼模型客户端 ======================
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY", "sk-your-api-key-here"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ====================== 安全组件 ======================
class CircuitBreaker:
    """熔断器：防止服务雪崩"""
    def __init__(self, failure_threshold=5, recovery_timeout=30):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.last_failure_time = 0

    def execute(self, func, *args, **kwargs):
        if self.failure_count >= self.failure_threshold:
            if time.time() - self.last_failure_time < self.recovery_timeout:
                raise Exception("服务熔断保护中，暂时不可用")
            self.failure_count = 0
        try:
            result = func(*args, **kwargs)
            self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            raise e

class ErrorHandler:
    """统一错误处理器"""
    def handle(self, error: Exception, context: Dict = None) -> str:
        error_type = self._classify(error)
        handlers = {
            "network": "❌ 网络不稳定，请检查后重试",
            "parameter": "❌ 参数错误，请检查输入格式",
            "authentication": "❌ API认证失败，请检查配置",
            "resource": "❌ 资源不足，请缩小图片后重试",
            "unknown": "❌ 系统繁忙，请稍后再试"
        }
        return handlers.get(error_type, handlers["unknown"])

    def _classify(self, error: Exception) -> str:
        msg = str(error).lower()
        if any(w in msg for w in ["timeout", "connection", "network"]):
            return "network"
        if any(w in msg for w in ["parameter", "invalid", "missing"]):
            return "parameter"
        if any(w in msg for w in ["auth", "key"]):
            return "authentication"
        if any(w in msg for w in ["memory", "quota", "limit"]):
            return "resource"
        return "unknown"

class ExponentialBackoffRetry:
    """指数退避重试"""
    def __init__(self, max_retries=2):
        self.max_retries = max_retries

    def retry(self, func, *args, **kwargs) -> Any:
        last_err = None
        for i in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_err = e
                if not self._is_retryable(e) or i >= self.max_retries:
                    break
                delay = min(2**i + random.uniform(0, 1), 5)
                time.sleep(delay)
        raise last_err

    def _is_retryable(self, e: Exception) -> bool:
        msg = str(e).lower()
        return any(w in msg for w in ["timeout", "connection", "busy"])

# ====================== 梦境配置 ======================
DREAM_CONFIGS = {
    "sweet": {
        "max_size": 512,
        "octave_layers": 3,
        "octave_scale": 0.9,
        "iter_per_octave": 30,
        "step_size": 0.006,
        "dream_softness": 1.2,
        "reality_fusion": 0.2,
        "dream_blur": True,
    },
    "fantasy": {
        "max_size": 512,
        "octave_layers": 4,
        "octave_scale": 0.85,
        "iter_per_octave": 40,
        "step_size": 0.008,
        "dream_softness": 1.0,
        "reality_fusion": 0.1,
        "dream_blur": True,
    },
    "nightmare": {
        "max_size": 512,
        "octave_layers": 3,
        "octave_scale": 0.8,
        "iter_per_octave": 50,
        "step_size": 0.01,
        "dream_softness": 0.8,
        "reality_fusion": 0.05,
        "dream_blur": False,
    },
    "scifi": {
        "max_size": 640,
        "octave_layers": 4,
        "octave_scale": 0.75,
        "iter_per_octave": 60,
        "step_size": 0.012,
        "dream_softness": 1.1,
        "reality_fusion": 0.0,
        "dream_blur": False,
    },
    "adventure": {
        "max_size": 512,
        "octave_layers": 3,
        "octave_scale": 0.9,
        "iter_per_octave": 35,
        "step_size": 0.007,
        "dream_softness": 1.3,
        "reality_fusion": 0.15,
        "dream_blur": True,
    }
}

# ====================== 结构化响应模式 ======================
class WeatherStructuredResponse(BaseModel):
    """天气查询结构化输出"""
    response_type: str = Field(default="weather", description="固定标识")
    punny_response: str = Field(description="双关语回复", min_length=2)
    city: str = Field(description="查询的城市名称")
    weather_info: str = Field(description="天气文本结果")

    @field_validator("response_type")
    def validate_type(cls, v):
        if v != "weather":
            raise ValueError("response_type必须为weather")
        return v

class DreamStructuredResponse(BaseModel):
    """梦境生成结构化输出"""
    response_type: str = Field(default="dream")
    punny_response: str = Field(description="双关语回复")
    dream_type: str = Field(description="梦境类型")
    user_uploaded_image: str = Field(description="用户上传的本地图片路径/文件名")
    output_image: str = Field(description="生成后的梦境图片保存路径")
    success: bool = Field(description="执行状态")

# ====================== 百炼模型封装 ======================
class BaiLianEmbeddings(Embeddings):
    def __init__(self, model="text-embedding-v3"):
        self.model = model
        self.client = client

    def embed_query(self, text: str) -> List[float]:
        try:
            response = self.client.embeddings.create(
                input=[text],
                model=self.model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Embedding查询失败: {e}")
            return [0.0] * 1536  # 返回默认向量

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model
            )
            return [emb.embedding for emb in response.data]
        except Exception as e:
            print(f"文档Embedding失败: {e}")
            return [[0.0] * 1536 for _ in texts]

class BaiLianChatModel(BaseChatModel):
    model_name: str = "qwen2.5-72b-instruct"
    temperature: float = 0.7

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = client
        self.retry = ExponentialBackoffRetry()
        self.breaker = CircuitBreaker()

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: List[str] = None,
        run_manager: CallbackManagerForLLMRun = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            def api_call():
                openai_messages = []
                for msg in messages:
                    if msg.type == "human":
                        openai_messages.append({"role": "user", "content": msg.content})
                    elif msg.type == "system":
                        openai_messages.append({"role": "system", "content": msg.content})
                    elif msg.type == "ai":
                        openai_messages.append({"role": "assistant", "content": msg.content})
                
                return self.client.chat.completions.create(
                    model=self.model_name,
                    messages=openai_messages,
                    temperature=self.temperature,
                    **kwargs
                )
            
            response = self.breaker.execute(lambda: self.retry.retry(api_call))
            
            message = AIMessage(content=response.choices[0].message.content)
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
            
        except Exception as e:
            error_msg = f"模型调用失败: {str(e)}"
            message = AIMessage(content=error_msg)
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: List[str] = None,
        run_manager: CallbackManagerForLLMRun = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop, run_manager, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "bailian"

# ====================== 梦境生成核心功能 ======================
def load_and_preprocess_image(image_path: str, max_size: int = 512) -> tf.Tensor:
    """加载和预处理图片"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片不存在: {image_path}")
    
    # 加载图片
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img, dtype=np.float32)
    
    # 调整大小
    h, w = img_np.shape[:2]
    max_side = max(h, w)
    if max_side > max_size:
        scale = max_size / max_side
        new_h, new_w = int(h * scale), int(w * scale)
        img_np = tf.image.resize(img_np, [new_h, new_w]).numpy()
    
    # 预处理
    img_tensor = tf.convert_to_tensor(img_np)
    img_tensor = tf.keras.applications.inception_v3.preprocess_input(img_tensor)
    
    return img_tensor

def build_dream_model(dream_type: str = "sweet"):
    """构建梦境生成模型"""
    base_model = tf.keras.applications.InceptionV3(
        include_top=False, 
        weights="imagenet"
    )
    
    # 根据梦境类型选择不同的层
    if dream_type == "sweet":
        layers = [base_model.get_layer("mixed0").output]
    elif dream_type == "fantasy":
        layers = [base_model.get_layer("mixed1").output, base_model.get_layer("mixed2").output]
    elif dream_type == "nightmare":
        layers = [base_model.get_layer("mixed3").output, base_model.get_layer("mixed4").output]
    elif dream_type == "scifi":
        layers = [base_model.get_layer("mixed5").output, base_model.get_layer("mixed6").output]
    elif dream_type == "adventure":
        layers = [base_model.get_layer("mixed0").output, base_model.get_layer("mixed1").output]
    else:
        layers = [base_model.get_layer("mixed0").output]
    
    return tf.keras.Model(inputs=base_model.input, outputs=layers)

@tf.function
def dream_step(image, model, step_size, grad_power=1.0, reality_fusion=0.1, original=None):
    """单个梦境生成步骤"""
    with tf.GradientTape() as tape:
        tape.watch(image)
        activations = model(tf.expand_dims(image, 0))
        loss = tf.reduce_sum([tf.reduce_mean(act) for act in activations])
    
    grads = tape.gradient(loss, image)
    grads /= tf.math.reduce_std(grads) + 1e-8
    
    if grad_power != 1.0:
        grads = grads * grad_power
    
    image += grads * step_size
    
    # 与原始图片融合
    if original is not None and reality_fusion > 0:
        image = image * (1 - reality_fusion) + original * reality_fusion
    
    return tf.clip_by_value(image, -1, 1)

def generate_deep_dream(
    image: tf.Tensor,
    dream_type: str = "sweet",
    max_size: int = 512,
    octave_layers: int = 3,
    octave_scale: float = 0.9,
    iter_per_octave: int = 30,
    step_size: float = 0.006,
    dream_softness: float = 1.0,
    reality_fusion: float = 0.1,
    dream_blur: bool = True
) -> tf.Tensor:
    """生成深度梦境"""
    # 构建模型
    model = build_dream_model(dream_type)
    
    # 多尺度处理
    original_shape = tf.shape(image)[:2]
    current = image
    
    # 生成多尺度图片
    for octave in range(octave_layers):
        scale = octave_scale ** (octave_layers - octave - 1)
        new_size = tf.cast(tf.cast(original_shape, tf.float32) * scale, tf.int32)
        
        # 调整大小
        current = tf.image.resize(current, new_size)
        original_scaled = tf.image.resize(image, new_size)
        
        # 在当前尺度上迭代
        for i in range(iter_per_octave):
            current = dream_step(
                current, model, step_size, 
                grad_power=dream_softness,
                reality_fusion=reality_fusion,
                original=original_scaled
            )
    
    # 调整回原始大小
    result = tf.image.resize(current, original_shape)
    
    return result

def postprocess_dream_image(tensor: tf.Tensor, dream_type: str) -> Image.Image:
    """后处理梦境图片"""
    # 转换为numpy数组
    img_np = tensor.numpy()
    
    # 反预处理
    img_np = (img_np + 1.0) * 127.5
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    
    # 创建PIL图片
    img = Image.fromarray(img_np)
    
    # 根据梦境类型应用不同效果
    if dream_type == "sweet":
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        # 稍微增加饱和度
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.2)
    elif dream_type == "nightmare":
        # 增加对比度
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.3)
    elif dream_type == "scifi":
        # 添加科技感色调
        r, g, b = img.split()
        b = b.point(lambda x: min(x + 20, 255))
        img = Image.merge("RGB", (r, g, b))
    
    return img

# 尝试导入ImageEnhance，如果失败则定义空函数
try:
    from PIL import ImageEnhance
except ImportError:
    class ImageEnhance:
        class Color:
            def __init__(self, img):
                self.img = img
            def enhance(self, factor):
                return self.img
        class Contrast:
            def __init__(self, img):
                self.img = img
            def enhance(self, factor):
                return self.img

# ====================== 工具函数 ======================
@tool
def get_weather(city: str) -> str:
    """查询城市天气"""
    try:
        # 模拟天气数据
        weather_data = {
            "北京": {"temp": "15-25°C", "condition": "晴", "desc": "适合做梦的好天气！"},
            "上海": {"temp": "18-22°C", "condition": "小雨", "desc": "梦里听雨声~"},
            "广州": {"temp": "25-32°C", "condition": "晴朗", "desc": "热带梦境在等你！"},
            "深圳": {"temp": "24-30°C", "condition": "多云", "desc": "科技感梦境启动！"},
            "杭州": {"temp": "16-24°C", "condition": "多云转晴", "desc": "西湖边的梦境~"},
            "成都": {"temp": "14-22°C", "condition": "阴", "desc": "火锅与梦境的结合！"}
        }
        
        if city in weather_data:
            data = weather_data[city]
            response = json.dumps({
                "response_type": "weather",
                "punny_response": f"🌤️ 为你查询{city}的天气！",
                "city": city,
                "weather_info": f"{city}：{data['condition']}，{data['temp']}，{data['desc']}"
            })
        else:
            response = json.dumps({
                "response_type": "weather",
                "punny_response": f"🌍 为你查询{city}的天气！",
                "city": city,
                "weather_info": f"{city}：天气数据获取中，先做个美梦吧~"
            })
        
        return response
        
    except Exception as e:
        return json.dumps({
            "response_type": "weather",
            "punny_response": "❌ 天气查询失败",
            "city": city,
            "weather_info": f"查询失败：{str(e)}"
        })

@tool
def generate_dream(
    dream_type: str = "sweet",
    user_image_path: str = "input.png"
) -> str:
    """
    根据用户上传的图片生成梦境
    
    Args:
        dream_type: 梦境类型，可选值：sweet, fantasy, nightmare, scifi, adventure
        user_image_path: 用户上传的图片路径
    """
    try:
        if not os.path.exists(user_image_path):
            return json.dumps({
                "response_type": "dream",
                "punny_response": f"❌ 未找到图片：{user_image_path}",
                "dream_type": dream_type,
                "user_uploaded_image": user_image_path,
                "output_image": "",
                "success": False
            })
        
        # 验证梦境类型
        if dream_type not in DREAM_CONFIGS:
            dream_type = "sweet"
        
        # 获取配置
        config = DREAM_CONFIGS[dream_type]
        
        # 确保输出目录存在
        os.makedirs("./dream_outputs", exist_ok=True)
        
        # 生成输出路径
        base_name = os.path.splitext(os.path.basename(user_image_path))[0]
        output_path = f"./dream_outputs/dream_{base_name}_{dream_type}_{int(time.time())}.png"
        
        try:
            # 1. 加载和预处理图片
            print(f"加载图片: {user_image_path}")
            img_tensor = load_and_preprocess_image(
                user_image_path, 
                max_size=config["max_size"]
            )
            
            # 2. 生成梦境
            print(f"生成{dream_type}梦境...")
            dream_tensor = generate_deep_dream(
                image=img_tensor,
                dream_type=dream_type,
                max_size=config["max_size"],
                octave_layers=config["octave_layers"],
                octave_scale=config["octave_scale"],
                iter_per_octave=config["iter_per_octave"],
                step_size=config["step_size"],
                dream_softness=config["dream_softness"],
                reality_fusion=config["reality_fusion"],
                dream_blur=config["dream_blur"]
            )
            
            # 3. 后处理
            print("后处理梦境图片...")
            dream_img = postprocess_dream_image(dream_tensor, dream_type)
            
            # 4. 保存结果
            print(f"保存梦境图片: {output_path}")
            dream_img.save(output_path)
            
            # 5. 创建对比图（可选）
            try:
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                
                # 原始图片
                original_img = Image.open(user_image_path).convert("RGB")
                axes[0].imshow(original_img)
                axes[0].set_title("原始图片", fontsize=14)
                axes[0].axis('off')
                
                # 梦境图片
                axes[1].imshow(dream_img)
                axes[1].set_title(f"{dream_type}梦境", fontsize=14)
                axes[1].axis('off')
                
                plt.tight_layout()
                comparison_path = output_path.replace(".png", "_comparison.png")
                plt.savefig(comparison_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                print(f"创建对比图失败: {e}")
            
            dream_types_cn = {
                "sweet": "甜蜜",
                "fantasy": "奇幻", 
                "nightmare": "噩梦",
                "scifi": "科幻",
                "adventure": "冒险"
            }
            
            return json.dumps({
                "response_type": "dream",
                "punny_response": f"✨ {dream_types_cn.get(dream_type, '美妙')}梦境已生成！",
                "dream_type": dream_types_cn.get(dream_type, "未知"),
                "user_uploaded_image": user_image_path,
                "output_image": output_path,
                "success": True
            })
            
        except Exception as e:
            print(f"梦境生成失败: {e}")
            # 如果生成失败，返回原始图片的简单处理版本
            try:
                original_img = Image.open(user_image_path).convert("RGB")
                # 应用简单的滤镜
                if dream_type == "sweet":
                    processed_img = original_img.filter(ImageFilter.GaussianBlur(1))
                elif dream_type == "nightmare":
                    processed_img = original_img.point(lambda x: x * 0.8)
                else:
                    processed_img = original_img
                
                processed_img.save(output_path)
                
                return json.dumps({
                    "response_type": "dream",
                    "punny_response": f"✨ 简易梦境已生成！",
                    "dream_type": dream_type,
                    "user_uploaded_image": user_image_path,
                    "output_image": output_path,
                    "success": True
                })
            except Exception as e2:
                return json.dumps({
                    "response_type": "dream",
                    "punny_response": f"❌ 梦境生成失败：{str(e)}",
                    "dream_type": dream_type,
                    "user_uploaded_image": user_image_path,
                    "output_image": "",
                    "success": False
                })
                
    except Exception as e:
        return json.dumps({
            "response_type": "dream",
            "punny_response": f"❌ 系统错误：{str(e)}",
            "dream_type": dream_type,
            "user_uploaded_image": user_image_path,
            "output_image": "",
            "success": False
        })

# 工具列表
TOOLS = [generate_dream, get_weather]

# ====================== 系统提示词 ======================
SYSTEM_PROMPT = """你是专注于【梦境图片生成】的智能助手，天气查询是附加功能。

核心功能：
1. 用户上传图片后，调用 generate_dream 工具生成梦境
2. 可查询城市天气（附加功能）

可用梦境类型：
- sweet: 甜蜜梦境（柔和、温暖）
- fantasy: 奇幻梦境（梦幻、神秘）
- nightmare: 噩梦（强烈、对比度高）
- scifi: 科幻梦境（冷色调、科技感）
- adventure: 冒险梦境（生动、有活力）

使用规则：
1. 用户上传图片后，询问或自动确定梦境类型
2. 调用 generate_dream 工具生成梦境
3. 可查询天气信息（附加功能）
4. 用中文回复，保持友好幽默
5. 如果用户没有指定梦境类型，询问或使用默认的"sweet"

工具参数说明：
- generate_dream: dream_type（梦境类型）, user_image_path（图片路径）
- get_weather: city（城市名）

响应格式：使用结构化JSON响应，包含必要信息。
"""

# ====================== 创建 Agent ======================
def create_agent():
    """创建并配置Agent"""
    try:
        # 创建模型
        llm = BaiLianChatModel()
        
        # 创建提示词模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad", optional=True),
        ])
        
        # 创建Agent
        agent = create_openai_tools_agent(
            llm=llm,
            tools=TOOLS,
            prompt=prompt
        )
        
        # 创建执行器
        agent_executor = AgentExecutor(
            agent=agent,
            tools=TOOLS,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3,
            early_stopping_method="generate"
        )
        
        return agent_executor
        
    except Exception as e:
        print(f"创建Agent失败: {e}")
        raise

# 初始化Agent
try:
    agent_executor = create_agent()
    print("✅ Agent初始化成功")
except Exception as e:
    print(f"❌ Agent初始化失败: {e}")
    agent_executor = None

# ====================== Web GUI ======================
def save_uploaded_image(image) -> str:
    """保存上传的图片并返回路径"""
    if image is None:
        return None
    
    # 确保目录存在
    os.makedirs("./uploads", exist_ok=True)
    
    # 生成唯一文件名
    filename = f"upload_{uuid.uuid4().hex[:8]}.png"
    filepath = os.path.join("./uploads", filename)
    
    # 保存图片
    try:
        if isinstance(image, str):  # 文件路径
            with open(image, 'rb') as src:
                with open(filepath, 'wb') as dst:
                    dst.write(src.read())
        else:  # PIL Image
            image.save(filepath)
        
        print(f"✅ 图片保存到: {filepath}")
        return filepath
    except Exception as e:
        print(f"❌ 保存图片失败: {e}")
        return None

def parse_agent_response(response_text: str) -> str:
    """解析Agent的响应"""
    try:
        # 尝试解析JSON
        if "response_type" in response_text:
            data = json.loads(response_text)
            
            if data.get("response_type") == "dream":
                if data.get("success"):
                    result = (
                        f"🎨 {data.get('punny_response', '梦境生成完成！')}\n"
                        f"📁 原图位置：{data.get('user_uploaded_image')}\n"
                        f"✨ 梦境图片：{data.get('output_image')}\n"
                        f"📂 梦境类型：{data.get('dream_type', '未知')}\n"
                        f"✅ 状态：成功生成"
                    )
                    
                    # 如果图片存在，添加预览信息
                    output_path = data.get("output_image")
                    if output_path and os.path.exists(output_path):
                        result += f"\n📸 图片已保存，可在文件管理器中查看"
                        
                else:
                    result = f"❌ {data.get('punny_response', '生成失败')}"
                    
            elif data.get("response_type") == "weather":
                result = f"🌤️ {data.get('punny_response', '')}\n{data.get('weather_info', '')}"
                
            else:
                result = response_text
        else:
            result = response_text
            
        return result
        
    except json.JSONDecodeError:
        # 如果不是JSON，直接返回
        return response_text
    except Exception as e:
        return f"❌ 解析响应失败：{str(e)}\n原始响应：{response_text}"

def agent_gui_chat(user_text: str, user_image):
    """
    GUI交互函数
    """
    if agent_executor is None:
        return "❌ Agent未初始化，请检查API密钥和网络连接"
    
    # 保存图片
    image_path = save_uploaded_image(user_image) if user_image else None
    
    # 构建输入
    input_text = user_text
    if image_path:
        input_text = f"{user_text} [图片路径：{image_path}]"
    
    # 准备输入
    inputs = {
        "input": input_text,
        "chat_history": []
    }
    
    # 执行Agent
    try:
        print(f"🤖 处理请求: {input_text[:100]}...")
        
        # 添加超时处理
        import threading
        from queue import Queue
        
        result_queue = Queue()
        
        def run_agent():
            try:
                response = agent_executor.invoke(inputs)
                result_queue.put(response)
            except Exception as e:
                result_queue.put({"error": str(e)})
        
        # 启动线程
        thread = threading.Thread(target=run_agent)
        thread.daemon = True
        thread.start()
        thread.join(timeout=30)  # 30秒超时
        
        if thread.is_alive():
            return "❌ 处理超时，请稍后重试"
        
        if result_queue.empty():
            return "❌ 未收到响应"
        
        response = result_queue.get()
        
        if "error" in response:
            return f"❌ 执行出错：{response['error']}"
        
        output_text = response.get("output", "抱歉，我没有得到有效的回复。")
        
        # 解析响应
        result = parse_agent_response(output_text)
        
        return result
        
    except Exception as e:
        error_msg = f"❌ 执行出错：{str(e)}"
        print(error_msg)
        return error_msg

# ====================== 清理函数 ======================
def cleanup_old_files(directory: str, max_age_hours: int = 24):
    """清理旧文件"""
    try:
        now = time.time()
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                file_age = now - os.path.getmtime(filepath)
                if file_age > max_age_hours * 3600:  # 转换为秒
                    os.remove(filepath)
                    print(f"清理文件: {filepath}")
    except Exception as e:
        print(f"清理文件失败: {e}")

# ====================== 启动界面 ======================
if __name__ == "__main__":
    # 创建必要的目录
    os.makedirs("./uploads", exist_ok=True)
    os.makedirs("./dream_outputs", exist_ok=True)
    
    # 清理旧文件
    cleanup_old_files("./uploads")
    cleanup_old_files("./dream_outputs")
    
    with gr.Blocks(title="梦境生成 Agent", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🌙 图片生成梦境智能助手
        **核心功能**：上传图片 → 生成艺术梦境 | **附加功能**：天气查询
        
        ## 使用说明：
        1. 上传一张图片
        2. 输入指令，例如：
           - "为这张图片生成一个甜蜜的梦境"
           - "这是什么图片的奇幻梦境？"
           - "查询北京天气"
           - "生成科幻风格的梦境"
        3. 点击生成按钮
        4. 查看结果，梦境图片会保存在 dream_outputs 目录
        
        ## 支持的梦境类型：
        - 🍬 甜蜜 (sweet): 柔和温暖的梦境
        - 🧚 奇幻 (fantasy): 神秘梦幻的梦境  
        - 😱 噩梦 (nightmare): 强烈对比的梦境
        - 🚀 科幻 (scifi): 科技感十足的梦境
        - 🏔️ 冒险 (adventure): 生动活力的梦境
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                user_image = gr.Image(
                    label="📤 上传图片（生成梦境必选）",
                    type="pil",
                    height=300,
                    interactive=True
                )
                
                gr.Markdown("""
                **图片要求：**
                - 支持 JPG、PNG 格式
                - 建议尺寸：512x512 到 1024x1024
                - 文件大小：小于 5MB
                """)
                
            with gr.Column(scale=2):
                user_text = gr.Textbox(
                    label="💭 输入指令",
                    placeholder="例：为这张图片生成甜蜜梦境 或 查询北京天气",
                    lines=3
                )
                
                output = gr.Textbox(
                    label="🤖 AI 响应",
                    lines=10,
                    interactive=False
                )
                
                with gr.Row():
                    submit_btn = gr.Button("🚀 开始生成", variant="primary", scale=2)
                    clear_btn = gr.Button("🧹 清空", variant="secondary", scale=1)
        
        # 绑定事件
        submit_btn.click(
            fn=agent_gui_chat,
            inputs=[user_text, user_image],
            outputs=output
        )
        
        def clear_all():
            return None, "", ""
        
        clear_btn.click(
            fn=clear_all,
            inputs=[],
            outputs=[user_image, user_text, output]
        )
        
        # 示例
        gr.Examples(
            examples=[
                ["为这张图片生成一个奇幻的梦境", None],
                ["查询北京天气怎么样？", None],
                ["为这张图片创建一个科幻梦境", None],
                ["这是什么图片的甜蜜梦境？", None],
                ["生成冒险风格的梦境", None],
            ],
            inputs=[user_text, user_image],
            label="💡 使用示例（点击自动填充）"
        )
        
        # 状态信息
        gr.Markdown("""
        ---
        **状态信息：**
        - ✅ Agent已初始化
        - 📁 上传目录: ./uploads/
        - 🎨 梦境输出: ./dream_outputs/
        - 🤖 模型: 百炼大模型
        """)
    
    # 启动
    print("=" * 50)
    print("🚀 梦境生成Agent启动中...")
    print("📌 请访问: http://localhost:7860")
    print("📌 或: http://127.0.0.1:7860")
    print("=" * 50)
    
    try:
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            inbrowser=True,
            show_error=True
        )
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        print("💡 尝试使用其他端口:")
        print("   demo.launch(server_port=7861)")