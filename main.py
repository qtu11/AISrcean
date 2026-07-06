import sys
import os
import io
import json
import base64
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLineEdit, QLabel,
    QSizeGrip, QFrame, QScrollArea, QSizePolicy,
    QComboBox, QMessageBox, QDialog, QFormLayout,
    QGraphicsOpacityEffect
)
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QSize, QPoint, QTimer, QRect, QPropertyAnimation
)
from PyQt5.QtGui import (
    QFont, QColor, QPalette, QPixmap, QCursor
)

from PIL import ImageGrab, Image
import google.generativeai as genai
import requests

# ─────────────────────────────────────────────
#  CONFIG PATH
# ─────────────────────────────────────────────
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.json")

DEFAULT_CONFIG = {
    "api_key": "",
    "gemini_api_key": "",
    "groq_api_key": "",
    "zhipu_api_key": "",
    "model": "gemini-2.0-flash-lite",
}

AVAILABLE_MODELS = [
    # Gemini
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-1.5-flash-latest",
    # Groq
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    # ZhipuAI
    "glm-4.5",
    "glm-4.5-air",
    "glm-4.6",
    "glm-4.7",
    "glm-5",
    "glm-5-turbo",
]

MODEL_PROVIDERS = {
    # Gemini
    "gemini-2.5-flash": "gemini",
    "gemini-2.0-flash": "gemini",
    "gemini-2.0-flash-lite": "gemini",
    "gemini-1.5-flash": "gemini",
    "gemini-1.5-pro": "gemini",
    "gemini-1.5-flash-latest": "gemini",
    "gemini-pro-vision": "gemini",
    # Groq
    "llama-3.3-70b-versatile": "groq",
    "llama-3.1-8b-instant": "groq",
    # Zhipu
    "glm-4.5": "zhipu",
    "glm-4.5-air": "zhipu",
    "glm-4.6": "zhipu",
    "glm-4.7": "zhipu",
    "glm-5": "zhipu",
    "glm-5-turbo": "zhipu",
    "glm-4-air": "zhipu",
    "glm-4": "zhipu",
}

VISION_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-1.5-flash-latest",
    "gemini-pro-vision",
]

def get_provider(model_name: str) -> str:
    return MODEL_PROVIDERS.get(model_name, "gemini")

SYSTEM_PROMPT = """Bạn là một AI Assistant thông minh, phân tích hình ảnh/màn hình cực kỳ chính xác.

Khi được cung cấp ảnh chụp màn hình, hãy:
1. Quan sát KỸ LƯỠNG toàn bộ nội dung trong ảnh.
2. Phân tích CHI TIẾT từng phần tử, văn bản, code, câu hỏi.
3. Nếu có câu hỏi trắc nghiệm → xác định ĐÁP ÁN ĐÚNG và giải thích rõ TẠI SAO.
4. Nếu là code → giải thích lỗi, đưa ra fix hoàn chỉnh.
5. Nếu là text → tóm tắt và trả lời chính xác.
6. Luôn suy nghĩ từng bước (chain of thought) trước khi kết luận.
7. Trả lời bằng tiếng Việt, ngắn gọn, chính xác.

Độ chính xác là ưu tiên số 1. Không đoán mò."""


def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {**DEFAULT_CONFIG, **data}
        except Exception:
            pass
    return DEFAULT_CONFIG.copy()


def save_config(cfg: dict):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────
#  API CALLING HELPERS
# ─────────────────────────────────────────────
def call_gemini(api_key, model_name, prompt, image):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=SYSTEM_PROMPT,
    )
    parts = []
    if image:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        img_bytes = buf.read()
        parts.append({
            "inline_data": {
                "mime_type": "image/png",
                "data": base64.b64encode(img_bytes).decode(),
            }
        })
    parts.append(prompt)
    response = model.generate_content(
        parts,
        generation_config=genai.types.GenerationConfig(
            temperature=0.1,
            max_output_tokens=4096,
        ),
        request_options={"timeout": 20.0}
    )
    return response.text

def call_groq(api_key, model_name, prompt, image):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    if image:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        img_bytes = buf.read()
        img_b64 = base64.b64encode(img_bytes).decode()
        
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
        ]
    else:
        content = prompt

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content}
        ],
        "temperature": 0.1,
        "max_tokens": 4096
    }
    
    r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=20)
    if r.status_code == 200:
        return r.json()['choices'][0]['message']['content']
    else:
        err_msg = r.text
        try:
            err_msg = r.json()['error']['message']
        except:
            pass
        raise Exception(f"Groq API error ({r.status_code}): {err_msg}")

def call_zhipu(api_key, model_name, prompt, image):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    if image:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        img_bytes = buf.read()
        img_b64 = base64.b64encode(img_bytes).decode()
        
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
        ]
    else:
        content = prompt

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content}
        ],
        "temperature": 0.1,
        "max_tokens": 4096
    }
    
    r = requests.post("https://open.bigmodel.cn/api/paas/v4/chat/completions", headers=headers, json=payload, timeout=20)
    if r.status_code == 200:
        return r.json()['choices'][0]['message']['content']
    else:
        err_msg = r.text
        try:
            err_msg = r.json()['error']['message']
        except:
            pass
        raise Exception(f"ZhipuAI API error ({r.status_code}): {err_msg}")


# ─────────────────────────────────────────────
#  UNIFIED LLM WORKER (WITH FAILOVER)
# ─────────────────────────────────────────────
class GeminiWorker(QThread):
    result_signal = pyqtSignal(str)
    error_signal  = pyqtSignal(str)
    status_signal = pyqtSignal(str)

    def __init__(self, cfg: dict, prompt: str, image: Image.Image = None):
        super().__init__()
        self.cfg    = cfg
        self.prompt = prompt
        self.image  = image

    def run(self):
        # 1. Xác định model chính và API keys
        main_model = self.cfg.get("model", "gemini-2.0-flash-lite")
        gemini_key = self.cfg.get("gemini_api_key", self.cfg.get("api_key", "")).strip()
        groq_key   = self.cfg.get("groq_api_key", "").strip()
        zhipu_key  = self.cfg.get("zhipu_api_key", "").strip()

        # 2. Xây dựng hàng đợi failover
        queue = []
        
        main_provider = get_provider(main_model)
        main_key = ""
        if main_provider == "gemini":
            main_key = gemini_key
        elif main_provider == "groq":
            main_key = groq_key
        elif main_provider == "zhipu":
            main_key = zhipu_key
            
        if main_key:
            queue.append((main_model, main_provider, main_key))

        is_vision = self.image is not None
        backup_models = []
        if is_vision:
            # Đối với hình ảnh: Chỉ dùng model hỗ trợ Vision (ưu tiên Gemini)
            backup_models = [
                ("gemini-2.5-flash", "gemini", gemini_key),
                ("gemini-2.0-flash", "gemini", gemini_key),
                ("gemini-2.0-flash-lite", "gemini", gemini_key),
                ("gemini-1.5-flash", "gemini", gemini_key),
                ("gemini-1.5-pro", "gemini", gemini_key),
            ]
        else:
            # Đối với text: Thử Groq trước vì ổn định nhất, rồi tới Gemini và Zhipu
            backup_models = [
                ("llama-3.3-70b-versatile", "groq", groq_key),
                ("llama-3.1-8b-instant", "groq", groq_key),
                ("gemini-2.5-flash", "gemini", gemini_key),
                ("gemini-2.0-flash", "gemini", gemini_key),
                ("gemini-2.0-flash-lite", "gemini", gemini_key),
                ("gemini-1.5-flash", "gemini", gemini_key),
                ("gemini-1.5-pro", "gemini", gemini_key),
                ("glm-4.5", "zhipu", zhipu_key),
                ("glm-4.5-air", "zhipu", zhipu_key),
                ("glm-5", "zhipu", zhipu_key),
                ("glm-5-turbo", "zhipu", zhipu_key),
            ]

        for m_name, prov, key in backup_models:
            if m_name != main_model and key:
                queue.append((m_name, prov, key))

        if not queue:
            self.error_signal.emit("❌ Lỗi: Không cấu hình API Key hợp lệ cho model được chọn hoặc model dự phòng!")
            return

        # 3. Thử tuần tự các model
        last_error = ""
        for i, (model_name, provider, api_key) in enumerate(queue):
            if i > 0:
                self.status_signal.emit(f"⚠️ Chuyển sang model dự phòng: {model_name}...")
            
            try:
                result_text = ""
                if provider == "gemini":
                    result_text = call_gemini(api_key, model_name, self.prompt, self.image)
                elif provider == "groq":
                    result_text = call_groq(api_key, model_name, self.prompt, self.image)
                elif provider == "zhipu":
                    result_text = call_zhipu(api_key, model_name, self.prompt, self.image)
                
                # Trả về kết quả kèm tên model thành công
                self.result_signal.emit(f"[{model_name}]{result_text}")
                return
            except Exception as e:
                err_str = str(e)
                print(f"Model {model_name} failed: {err_str}")
                last_error = f"Model {model_name} lỗi: {err_str}"

        # Tất cả đều thất bại
        self.error_signal.emit(f"❌ Tất cả các model đều thất bại. Lỗi cuối cùng:\n{last_error}")


# ─────────────────────────────────────────────
#  CUSTOM HEADER FRAME
# ─────────────────────────────────────────────
class HeaderFrame(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.assistant = parent

    def enterEvent(self, e):
        # Hiện ký hiệu điều khiển khi rê chuột vào thanh tiêu đề
        if hasattr(self.assistant, "btn_close") and self.assistant.btn_close:
            self.assistant.btn_close.setText("✕")
            self.assistant.btn_close.setStyleSheet("background: #ef4444; color: #7f1d1d; border: none; border-radius: 6px; font-weight: bold; font-size: 7px;")
        if hasattr(self.assistant, "btn_collapse") and self.assistant.btn_collapse:
            self.assistant.btn_collapse.setText("▲" if self.assistant._collapsed else "▼")
            self.assistant.btn_collapse.setStyleSheet("background: #f59e0b; color: #78350f; border: none; border-radius: 6px; font-weight: bold; font-size: 7px;")
        if hasattr(self.assistant, "btn_settings") and self.assistant.btn_settings:
            self.assistant.btn_settings.setText("⚙")
            self.assistant.btn_settings.setStyleSheet("background: #10b981; color: #064e3b; border: none; border-radius: 6px; font-weight: bold; font-size: 8px;")
        super().enterEvent(e)

    def leaveEvent(self, e):
        # Trở về trạng thái chấm tròn màu thuần khi rời đi
        if hasattr(self.assistant, "btn_close") and self.assistant.btn_close:
            self.assistant.btn_close.setText("")
            self.assistant.btn_close.setStyleSheet("background: #ef4444; color: transparent; border: none; border-radius: 6px;")
        if hasattr(self.assistant, "btn_collapse") and self.assistant.btn_collapse:
            self.assistant.btn_collapse.setText("")
            self.assistant.btn_collapse.setStyleSheet("background: #f59e0b; color: transparent; border: none; border-radius: 6px;")
        if hasattr(self.assistant, "btn_settings") and self.assistant.btn_settings:
            self.assistant.btn_settings.setText("")
            self.assistant.btn_settings.setStyleSheet("background: #10b981; color: transparent; border: none; border-radius: 6px;")
        super().leaveEvent(e)


# ─────────────────────────────────────────────
#  CHAT BUBBLE
# ─────────────────────────────────────────────
class ChatBubble(QFrame):
    def __init__(self, text: str, is_user: bool, model_name: str = None, parent=None):
        super().__init__(parent)
        self.is_user = is_user
        
        # Opacity effect for smooth fade-in
        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)
        
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(8)
        
        # Rounded Avatars
        self.avatar = QLabel()
        self.avatar.setFixedSize(28, 28)
        self.avatar.setAlignment(Qt.AlignCenter)
        self.avatar.setFont(QFont("Segoe UI", 9, QFont.Bold))
        
        if is_user:
            self.avatar.setText("U")
            self.avatar.setStyleSheet("""
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #8b5cf6, stop:1 #ec4899);
                color: white;
                border-radius: 14px;
            """)
        else:
            self.avatar.setText("AI")
            self.avatar.setStyleSheet("""
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #10b981, stop:1 #06b6d4);
                color: white;
                border-radius: 14px;
            """)
            
        # Message content box
        self.content_frame = QFrame()
        self.content_frame.setObjectName("content_frame")
        content_layout = QVBoxLayout(self.content_frame)
        content_layout.setContentsMargins(12, 8, 12, 8)
        content_layout.setSpacing(3)
        
        sender_text = "Bạn"
        if not is_user:
            sender_text = f"AI ({model_name})" if model_name else "AI"
            
        sender = QLabel(sender_text)
        sender.setFont(QFont("Segoe UI", 8, QFont.Bold))
        sender.setStyleSheet("color: #d8b4fe;" if is_user else "color: #34d399;")
        sender.setAttribute(Qt.WA_TranslucentBackground)
        content_layout.addWidget(sender)
        
        msg = QLabel(text)
        msg.setWordWrap(True)
        msg.setFont(QFont("Segoe UI", 9))
        msg.setTextInteractionFlags(Qt.TextSelectableByMouse)
        msg.setStyleSheet("color: #f1f5f9; background: transparent; border: none;")
        msg.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        content_layout.addWidget(msg)
        
        ts = QLabel(datetime.now().strftime("%H:%M:%S"))
        ts.setFont(QFont("Segoe UI", 7))
        ts.setStyleSheet("color: rgba(226, 232, 240, 0.35); background: transparent; border: none;")
        content_layout.addWidget(ts)
        
        if is_user:
            self.content_frame.setStyleSheet("""
                QFrame#content_frame {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 rgba(139, 92, 246, 0.2), stop:1 rgba(236, 72, 153, 0.1));
                    border: 1px solid rgba(139, 92, 246, 0.3);
                    border-top-left-radius: 12px;
                    border-top-right-radius: 12px;
                    border-bottom-left-radius: 12px;
                    border-bottom-right-radius: 2px;
                }
            """)
            main_layout.addWidget(self.content_frame, 1)
            main_layout.addWidget(self.avatar)
        else:
            self.content_frame.setStyleSheet("""
                QFrame#content_frame {
                    background: rgba(30, 41, 59, 0.6);
                    border: 1px solid rgba(16, 185, 129, 0.2);
                    border-top-left-radius: 12px;
                    border-top-right-radius: 12px;
                    border-bottom-right-radius: 12px;
                    border-bottom-left-radius: 2px;
                }
            """)
            main_layout.addWidget(self.avatar)
            main_layout.addWidget(self.content_frame, 1)
            
        self.setStyleSheet("background: transparent; border: none;")
        
        # Start fade-in animation
        self.anim = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.anim.setDuration(350)
        self.anim.setStartValue(0.0)
        self.anim.setEndValue(1.0)
        self.anim.start()


# ─────────────────────────────────────────────
#  SETTINGS DIALOG
# ─────────────────────────────────────────────
class SettingsDialog(QDialog):
    def __init__(self, cfg: dict, parent=None):
        super().__init__(parent)
        self.cfg = cfg.copy()
        self.setWindowTitle("⚙ Cài đặt API")
        self.setFixedSize(400, 280)
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #090d16, stop:1 #110b29);
                color: #f1f5f9;
                border: 1px solid rgba(139, 92, 246, 0.4);
                border-radius: 12px;
            }
            QLabel {
                color: #cbd5e1;
                font-size: 11px;
                font-weight: bold;
            }
            QLineEdit, QComboBox {
                background: rgba(15, 23, 42, 0.85);
                color: #f1f5f9;
                border: 1px solid rgba(139, 92, 246, 0.3);
                border-radius: 8px;
                padding: 6px 10px;
                font-size: 11px;
            }
            QLineEdit:focus, QComboBox:focus {
                border-color: #8b5cf6;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #8b5cf6, stop:1 #ec4899);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 18px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7c3aed, stop:1 #db2777);
            }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 16, 20, 16)

        form = QFormLayout()
        form.setSpacing(10)

        # Gemini Key
        self.gemini_key_input = QLineEdit(self.cfg.get("gemini_api_key", self.cfg.get("api_key", "")))
        self.gemini_key_input.setPlaceholderText("AIza...")
        self.gemini_key_input.setEchoMode(QLineEdit.Password)
        form.addRow("🔑 Gemini API Key:", self.gemini_key_input)

        # Groq Key
        self.groq_key_input = QLineEdit(self.cfg.get("groq_api_key", ""))
        self.groq_key_input.setPlaceholderText("gsk_...")
        self.groq_key_input.setEchoMode(QLineEdit.Password)
        form.addRow("🔑 Groq API Key:", self.groq_key_input)

        # Zhipu Key
        self.zhipu_key_input = QLineEdit(self.cfg.get("zhipu_api_key", ""))
        self.zhipu_key_input.setPlaceholderText("8fa2...")
        self.zhipu_key_input.setEchoMode(QLineEdit.Password)
        form.addRow("🔑 Zhipu API Key:", self.zhipu_key_input)

        # Model
        self.model_combo = QComboBox()
        for m in AVAILABLE_MODELS:
            self.model_combo.addItem(m)
        cur = self.cfg.get("model", AVAILABLE_MODELS[0])
        if cur in AVAILABLE_MODELS:
            self.model_combo.setCurrentText(cur)
        form.addRow("🤖 Model chính:", self.model_combo)

        layout.addLayout(form)

        hint = QLabel("💡 Lấy API Keys tại: aistudio.google.com | console.groq.com | bigmodel.cn")
        hint.setStyleSheet("color: #64748b; font-size: 9.5px;")
        layout.addWidget(hint)

        btn_row = QHBoxLayout()
        btn_save = QPushButton("💾 Lưu")
        btn_save.clicked.connect(self._save)
        btn_cancel = QPushButton("Hủy")
        btn_cancel.setStyleSheet("background: rgba(55, 65, 81, 0.85); color: #e2e8f0; border: none;")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_save)
        layout.addLayout(btn_row)

    def _save(self):
        self.cfg["gemini_api_key"] = self.gemini_key_input.text().strip()
        self.cfg["groq_api_key"]   = self.groq_key_input.text().strip()
        self.cfg["zhipu_api_key"]  = self.zhipu_key_input.text().strip()
        self.cfg["api_key"]        = self.cfg["gemini_api_key"]  # Tương thích ngược
        self.cfg["model"]          = self.model_combo.currentText()
        self.accept()

    def get_config(self):
        return self.cfg


# ─────────────────────────────────────────────
#  SCREENSHOT OVERLAY
# ─────────────────────────────────────────────
class ScreenshotOverlay(QWidget):
    captured = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setCursor(Qt.CrossCursor)
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(screen)
        self.origin = QPoint()
        self.end    = QPoint()
        self._drag  = False

    def paintEvent(self, _):
        from PyQt5.QtGui import QPainter, QPen
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(0, 0, 0, 110))
        if self._drag:
            sel = QRect(self.origin, self.end).normalized()
            p.fillRect(sel, QColor(0, 0, 0, 0))
            
            # Neon border dash line
            pen = QPen(QColor(139, 92, 246), 1.5, Qt.DashLine)
            p.setPen(pen)
            p.drawRect(sel)
            
            # Corner markers for design look
            p.setPen(QPen(QColor(236, 72, 153), 2.5, Qt.SolidLine))
            r = sel
            length = 10
            p.drawLine(r.left(), r.top(), r.left() + length, r.top())
            p.drawLine(r.left(), r.top(), r.left(), r.top() + length)
            
            p.drawLine(r.right(), r.top(), r.right() - length, r.top())
            p.drawLine(r.right(), r.top(), r.right(), r.top() + length)
            
            p.drawLine(r.left(), r.bottom(), r.left() + length, r.bottom())
            p.drawLine(r.left(), r.bottom(), r.left(), r.bottom() - length)
            
            p.drawLine(r.right(), r.bottom(), r.right() - length, r.bottom())
            p.drawLine(r.right(), r.bottom(), r.right(), r.bottom() - length)

    def mousePressEvent(self, e):
        self.origin = self.end = e.pos()
        self._drag = True

    def mouseMoveEvent(self, e):
        self.end = e.pos()
        self.update()

    def mouseReleaseEvent(self, e):
        self.end   = e.pos()
        self._drag = False
        self.hide()
        sel = QRect(self.origin, self.end).normalized()
        QTimer.singleShot(200, lambda: self._capture(sel if sel.width() > 10 else None))

    def _capture(self, rect):
        try:
            img = ImageGrab.grab(
                bbox=(rect.x(), rect.y(), rect.x() + rect.width(), rect.y() + rect.height())
                if rect else None
            )
            self.captured.emit(img)
        except Exception as ex:
            print(f"Screenshot error: {ex}")
            self.captured.emit(None)
        self.close()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.captured.emit(None)
            self.close()


# ─────────────────────────────────────────────
#  IMAGE PREVIEW WIDGET
# ─────────────────────────────────────────────
class ImagePreviewWidget(QFrame):
    clear_requested = pyqtSignal()
    take_screenshot_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(90)
        self.has_image = False
        self.pixmap = None

        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(6, 6, 6, 6)
        
        self.content_lbl = QLabel("📷 Chưa có ảnh — Click để chụp màn hình")
        self.content_lbl.setAlignment(Qt.AlignCenter)
        self.content_lbl.setFont(QFont("Segoe UI", 9))
        self.content_lbl.setStyleSheet("color: #64748b; background: transparent; border: none;")
        self.layout.addWidget(self.content_lbl)

        self.delete_btn = QPushButton("🗑 Xóa ảnh", self)
        self.delete_btn.setFixedSize(90, 26)
        self.delete_btn.setStyleSheet("""
            QPushButton {
                background: rgba(220, 38, 38, 0.85);
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                font-size: 10px;
            }
            QPushButton:hover {
                background: rgba(220, 38, 38, 0.95);
            }
        """)
        self.delete_btn.clicked.connect(self.clear_requested.emit)
        self.delete_btn.hide()
        
        self.update_style()

    def update_style(self):
        if self.has_image:
            self.setStyleSheet("""
                QFrame {
                    background: rgba(15, 23, 42, 0.6);
                    border: 1px solid rgba(139, 92, 246, 0.3);
                    border-radius: 12px;
                }
            """)
        else:
            self.setStyleSheet("""
                QFrame {
                    background: rgba(15, 23, 42, 0.3);
                    border: 1.5px dashed rgba(139, 92, 246, 0.2);
                    border-radius: 12px;
                }
                QFrame:hover {
                    background: rgba(15, 23, 42, 0.5);
                    border-color: rgba(139, 92, 246, 0.45);
                }
            """)

    def set_image(self, img_pixmap):
        self.pixmap = img_pixmap
        self.has_image = True
        self.content_lbl.setText("")
        scaled_px = img_pixmap.scaled(self.width() - 16, self.height() - 16, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.content_lbl.setPixmap(scaled_px)
        self.update_style()

    def clear_image(self):
        self.pixmap = None
        self.has_image = False
        self.content_lbl.setPixmap(QPixmap())
        self.content_lbl.setText("📷 Chưa có ảnh — Click để chụp màn hình")
        self.delete_btn.hide()
        self.update_style()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.delete_btn.move(
            (self.width() - self.delete_btn.width()) // 2,
            (self.height() - self.delete_btn.height()) // 2
        )
        if self.has_image and self.pixmap:
            scaled_px = self.pixmap.scaled(self.width() - 16, self.height() - 16, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.content_lbl.setPixmap(scaled_px)

    def enterEvent(self, e):
        if self.has_image:
            self.delete_btn.show()
        super().enterEvent(e)

    def leaveEvent(self, e):
        self.delete_btn.hide()
        super().leaveEvent(e)

    def mousePressEvent(self, e):
        if not self.has_image:
            self.take_screenshot_requested.emit()
        super().mousePressEvent(e)


# ─────────────────────────────────────────────
#  CUSTOM TEXT EDIT (AUTO-RESIZE)
# ─────────────────────────────────────────────
class CustomTextEdit(QTextEdit):
    returnPressed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFixedHeight(38)
        self.textChanged.connect(self.adjust_height)
        self.setFont(QFont("Segoe UI", 9))
        self.setPlaceholderText("Nhập câu hỏi hoặc nhấn 📸 để phân tích ảnh...")
        self.setStyleSheet("""
            QTextEdit {
                background: rgba(15, 23, 42, 0.85);
                color: #e2e8f0;
                border: 1px solid rgba(139, 92, 246, 0.4);
                border-radius: 12px;
                padding: 6px 10px;
            }
            QTextEdit:focus {
                border-color: #8b5cf6;
            }
        """)

    def adjust_height(self):
        doc_height = int(self.document().size().height())
        new_height = doc_height + 14
        new_height = max(38, min(new_height, 120))
        self.setFixedHeight(new_height)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if event.modifiers() & Qt.ShiftModifier:
                super().keyPressEvent(event)
            else:
                self.returnPressed.emit()
        else:
            super().keyPressEvent(event)

    def text(self):
        return self.toPlainText()

    def clear(self):
        self.setPlainText("")
        self.setFixedHeight(38)


# ─────────────────────────────────────────────
#  MAIN WINDOW
# ─────────────────────────────────────────────
class AIAssistant(QWidget):
    def __init__(self):
        super().__init__()
        self._cfg        = load_config()
        self._screenshot: Image.Image = None
        self._worker:     GeminiWorker = None
        self._drag_pos   = None
        self._collapsed  = False

        self._init_window()
        self._init_ui()
        self._position_window()

        # Startup animation (Fade-in)
        self.setWindowOpacity(0.0)
        self.fade_in = QPropertyAnimation(self, b"windowOpacity")
        self.fade_in.setDuration(350)
        self.fade_in.setStartValue(0.0)
        self.fade_in.setEndValue(0.96)
        self.fade_in.start()

        # Typing Animation Timer
        self.typing_timer = QTimer()
        self.typing_timer.setInterval(400)
        self.typing_timer.timeout.connect(self._animate_typing)
        self.typing_dots = 0

        # Nếu chưa có key → mở settings ngay
        if not self._cfg.get("api_key"):
            QTimer.singleShot(300, self._open_settings)

    def _init_window(self):
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setMinimumSize(360, 120)
        self.resize(385, 590)

    def _position_window(self):
        screen = QApplication.primaryScreen().geometry()
        self.move(screen.width() - self.width() - 16,
                  screen.height() - self.height() - 50)

    # ── UI ────────────────────────────────────
    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        self.card = QFrame()
        self.card.setObjectName("card")
        root.addWidget(self.card)
        cl = QVBoxLayout(self.card)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)

        # ── Header ─────────────────────────
        self.hdr = HeaderFrame(self)
        self.hdr.setObjectName("hdr")
        self.hdr.setFixedHeight(46)
        
        # Đặt style ban đầu cho header và card
        self.update_collapsed_style(False)
        
        hl = QHBoxLayout(self.hdr)
        hl.setContentsMargins(16, 0, 16, 0)

        lbl_icon = QLabel("✨")
        lbl_icon.setFont(QFont("Segoe UI Emoji", 12))
        lbl_icon.setStyleSheet("background: transparent; border: none;")
        hl.addWidget(lbl_icon)

        lbl_title = QLabel("AI Assistant")
        lbl_title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        lbl_title.setStyleSheet("color: #d8b4fe; background: transparent; border: none;")
        hl.addWidget(lbl_title)

        hl.addStretch()

        self.status_dot = QLabel("●")
        self.status_dot.setFont(QFont("Segoe UI", 10))
        self.status_dot.setStyleSheet(self._dot_style("#34d399"))
        hl.addWidget(self.status_dot)
        hl.addSpacing(8)

        # macOS style window control dots (được quản lý động bằng enter/leave event của HeaderFrame)
        self.btn_settings = QPushButton("")
        self.btn_settings.setFixedSize(12, 12)
        self.btn_settings.setToolTip("Cài đặt API Key & Model")
        self.btn_settings.setStyleSheet("background: #10b981; color: transparent; border: none; border-radius: 6px;")
        self.btn_settings.clicked.connect(self._open_settings)
        hl.addWidget(self.btn_settings)
        hl.addSpacing(4)

        self.btn_collapse = QPushButton("")
        self.btn_collapse.setFixedSize(12, 12)
        self.btn_collapse.setToolTip("Thu gọn / Mở rộng")
        self.btn_collapse.setStyleSheet("background: #f59e0b; color: transparent; border: none; border-radius: 6px;")
        self.btn_collapse.clicked.connect(self._toggle_collapse)
        hl.addWidget(self.btn_collapse)
        hl.addSpacing(4)

        self.btn_close = QPushButton("")
        self.btn_close.setFixedSize(12, 12)
        self.btn_close.setToolTip("Đóng")
        self.btn_close.setStyleSheet("background: #ef4444; color: transparent; border: none; border-radius: 6px;")
        self.btn_close.clicked.connect(self._close_app)
        hl.addWidget(self.btn_close)

        cl.addWidget(self.hdr)

        # ── Body ───────────────────────────
        self.body = QFrame()
        self.body.setStyleSheet("background: transparent;")
        bl = QVBoxLayout(self.body)
        bl.setContentsMargins(12, 10, 12, 12)
        bl.setSpacing(8)

        # Model label
        self.model_lbl = QLabel(f"Model: {self._cfg['model']}")
        self.model_lbl.setFont(QFont("Segoe UI", 8, QFont.Bold))
        self.model_lbl.setStyleSheet("color: rgba(226, 232, 240, 0.45);")
        bl.addWidget(self.model_lbl)

        # Custom Image Preview Widget
        self.image_preview = ImagePreviewWidget()
        self.image_preview.clear_requested.connect(self._clear_screenshot)
        self.image_preview.take_screenshot_requested.connect(self._take_screenshot)
        bl.addWidget(self.image_preview)

        # Action Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        
        self.btn_ss = QPushButton("📸  Chụp màn hình")
        self.btn_ss.setFixedHeight(34)
        self.btn_ss.setStyleSheet(self._action_btn_gradient())
        self.btn_ss.clicked.connect(self._take_screenshot)
        btn_row.addWidget(self.btn_ss)
        bl.addLayout(btn_row)

        # Chat area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumHeight(210)
        scroll.setStyleSheet("""
            QScrollArea {
                background: rgba(15, 23, 42, 0.4);
                border: 1px solid rgba(139, 92, 246, 0.25);
                border-radius: 12px;
            }
            QScrollBar:vertical {
                background: rgba(0, 0, 0, 0.2); 
                width: 4px; 
                border-radius: 2px;
                margin: 4px;
            }
            QScrollBar::handle:vertical {
                background: rgba(139, 92, 246, 0.7); 
                border-radius: 2px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(139, 92, 246, 0.95); 
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { 
                height: 0; 
            }
        """)
        self._scroll = scroll

        self.chat_container = QWidget()
        self.chat_container.setStyleSheet("background: transparent;")
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setContentsMargins(6, 6, 6, 6)
        self.chat_layout.setSpacing(8)
        self.chat_layout.addStretch()
        scroll.setWidget(self.chat_container)
        bl.addWidget(scroll)

        # Typing
        self.typing_lbl = QLabel("🤖 AI đang suy nghĩ...")
        self.typing_lbl.setFont(QFont("Segoe UI", 8, QFont.Bold))
        self.typing_lbl.setStyleSheet("color: #a78bfa; padding: 2px 4px;")
        self.typing_lbl.hide()
        bl.addWidget(self.typing_lbl)

        # Input Row
        inp_row = QHBoxLayout()
        inp_row.setSpacing(8)
        
        self.input_field = CustomTextEdit()
        self.input_field.returnPressed.connect(self._send)
        inp_row.addWidget(self.input_field, 1)

        self.btn_send = QPushButton("➤")
        self.btn_send.setFixedSize(38, 38)
        self.btn_send.setStyleSheet(self._send_btn_style())
        self.btn_send.clicked.connect(self._send)
        inp_row.addWidget(self.btn_send)
        bl.addLayout(inp_row)

        cl.addWidget(self.body)

        # Size grip
        gr = QHBoxLayout()
        gr.setContentsMargins(0, 0, 4, 4)
        gr.addStretch()
        grip = QSizeGrip(self)
        grip.setStyleSheet("background: transparent;")
        gr.addWidget(grip)
        cl.addLayout(gr)

    # ── Styles ────────────────────────────────
    def _dot_style(self, color):
        return f"color: {color}; background: transparent; border: none;"

    def _action_btn_gradient(self):
        return """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7c3aed, stop:1 #a855f7);
                color: white; 
                border: none;
                border-radius: 17px; 
                font-weight: bold; 
                font-size: 11px;
                padding: 0 16px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #6d28d9, stop:1 #9333ea);
            }
            QPushButton:pressed { 
                background: #5b21b6; 
            }
            QPushButton:disabled { 
                background: #374151; 
                color: #6b7280; 
            }
        """

    def _send_btn_style(self):
        return """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #8b5cf6, stop:1 #ec4899);
                color: white; 
                border: none;
                border-radius: 19px; 
                font-weight: bold; 
                font-size: 13px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #7c3aed, stop:1 #db2777);
            }
            QPushButton:pressed { 
                background: #4c1d95; 
            }
            QPushButton:disabled { 
                background: #374151; 
                color: #6b7280; 
            }
        """

    # ── Drag ──────────────────────────────────
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag_pos = e.globalPos() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        if e.buttons() == Qt.LeftButton and self._drag_pos:
            self.move(e.globalPos() - self._drag_pos)

    def mouseReleaseEvent(self, _):
        self._drag_pos = None

    # ── Update Style for Collapse/Expand ──
    def update_collapsed_style(self, collapsed):
        if collapsed:
            self.card.setStyleSheet("""
                QFrame#card {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #090d16, stop:1 #110b29);
                    border: 1px solid rgba(139, 92, 246, 0.55);
                    border-radius: 23px;
                }
            """)
            self.hdr.setStyleSheet("""
                QFrame#hdr {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1e1b4b, stop:1 #311068);
                    border-radius: 23px;
                    border: none;
                }
            """)
        else:
            self.card.setStyleSheet("""
                QFrame#card {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #090d16, stop:1 #110b29);
                    border: 1px solid rgba(139, 92, 246, 0.45);
                    border-radius: 16px;
                }
            """)
            self.hdr.setStyleSheet("""
                QFrame#hdr {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1e1b4b, stop:1 #311068);
                    border-radius: 16px 16px 0 0;
                    border-bottom: 1px solid rgba(139, 92, 246, 0.25);
                }
            """)

    # ── Collapse / Expand (Smooth Animation) ──
    def _toggle_collapse(self):
        self._collapsed = not self._collapsed
        
        start_size = self.size()
        end_size = QSize(385, 46) if self._collapsed else QSize(385, 590)
        
        # Cập nhật style bo tròn của card và hdr ngay lập tức
        self.update_collapsed_style(self._collapsed)
        
        if self._collapsed:
            # Ẩn body ngay lập tức để tránh layout chồng chéo trong lúc co lại
            self.body.setVisible(False)
            
        self.resize_anim = QPropertyAnimation(self, b"size")
        self.resize_anim.setDuration(260)
        self.resize_anim.setStartValue(start_size)
        self.resize_anim.setEndValue(end_size)
        
        if not self._collapsed:
            # Hiển thị lại body sau khi mở rộng xong
            self.resize_anim.finished.connect(lambda: self.body.setVisible(True))
        else:
            try:
                self.resize_anim.finished.disconnect()
            except:
                pass
                
        self.resize_anim.start()

    # ── Close Application (Smooth Fade-out) ──
    def _close_app(self):
        self.fade_out = QPropertyAnimation(self, b"windowOpacity")
        self.fade_out.setDuration(250)
        self.fade_out.setStartValue(self.windowOpacity())
        self.fade_out.setEndValue(0.0)
        self.fade_out.finished.connect(QApplication.quit)
        self.fade_out.start()

    # ── Settings ──────────────────────────────
    def _open_settings(self):
        dlg = SettingsDialog(self._cfg, self)
        if dlg.exec_() == QDialog.Accepted:
            self._cfg = dlg.get_config()
            save_config(self._cfg)
            self.model_lbl.setText(f"Model: {self._cfg['model']}")
            self._add_bubble(f"✅ Đã lưu! Model: {self._cfg['model']}", is_user=False)

    # ── Screenshot ────────────────────────────
    def _take_screenshot(self):
        self.hide()
        QTimer.singleShot(300, self._show_overlay)

    def _show_overlay(self):
        self._overlay = ScreenshotOverlay()
        self._overlay.captured.connect(self._on_captured)
        self._overlay.showFullScreen()

    def _on_captured(self, img):
        self.show()
        if img is None:
            return
        self._screenshot = img
        thumb = img.copy()
        thumb.thumbnail((220, 78), Image.LANCZOS)
        buf = io.BytesIO()
        thumb.save(buf, format="PNG")
        px = QPixmap()
        px.loadFromData(buf.getvalue())
        
        self.image_preview.set_image(px)
        self.status_dot.setStyleSheet(self._dot_style("#f59e0b"))

    def _clear_screenshot(self):
        self._screenshot = None
        self.image_preview.clear_image()
        self.status_dot.setStyleSheet(self._dot_style("#34d399"))

    # ── Send ──────────────────────────────────
    def _send(self):
        text = self.input_field.text().strip()
        if not text and self._screenshot is None:
            return

        gemini_key = self._cfg.get("gemini_api_key", self._cfg.get("api_key", "")).strip()
        groq_key   = self._cfg.get("groq_api_key", "").strip()
        zhipu_key  = self._cfg.get("zhipu_api_key", "").strip()
        
        if not gemini_key and not groq_key and not zhipu_key:
            self._add_bubble("⚠ Chưa cấu hình API Key nào! Nhấn nút cài đặt để nhập.", is_user=False)
            return

        prompt = text if text else "Phân tích chi tiết và đưa ra đáp án/kết quả chính xác nhất từ ảnh này."
        self._add_bubble(prompt if text else "📸 [Phân tích ảnh]", is_user=True)
        self.input_field.clear()

        self.btn_send.setEnabled(False)
        self.btn_ss.setEnabled(False)
        self.input_field.setEnabled(False)
        
        main_model = self._cfg["model"]
        self.typing_lbl.setText(f"🤖 AI ({main_model}) đang suy nghĩ...")
        self.typing_lbl.show()
        
        # Start typing animation timer
        self.typing_dots = 0
        self.typing_timer.start()
        
        self.status_dot.setStyleSheet(self._dot_style("#f59e0b"))

        self._worker = GeminiWorker(self._cfg, prompt, self._screenshot)
        self._worker.result_signal.connect(self._on_result)
        self._worker.error_signal.connect(self._on_error)
        self._worker.status_signal.connect(self._on_status_update)
        self._worker.start()

    def _animate_typing(self):
        self.typing_dots = (self.typing_dots + 1) % 4
        dots = "." * self.typing_dots
        main_model = self._cfg["model"]
        self.typing_lbl.setText(f"🤖 AI ({main_model}) đang suy nghĩ{dots}")

    def _on_status_update(self, msg):
        self.typing_timer.stop()
        self.typing_lbl.setText(msg)
        self.typing_lbl.show()

    def _on_result(self, text):
        self.typing_timer.stop()
        self.typing_lbl.hide()
        
        model_name = self._cfg["model"]
        content = text
        if text.startswith("[") and "]" in text:
            idx = text.find("]")
            model_name = text[1:idx]
            content = text[idx+1:]
            
        self._add_bubble(content, is_user=False, model_name=model_name)
        self._re_enable()
        self.status_dot.setStyleSheet(self._dot_style("#34d399"))

    def _on_error(self, err):
        self.typing_timer.stop()
        self.typing_lbl.hide()
        self._add_bubble(err, is_user=False)
        self._re_enable()
        self.status_dot.setStyleSheet(self._dot_style("#ef4444"))

    def _re_enable(self):
        self.btn_send.setEnabled(True)
        self.btn_ss.setEnabled(True)
        self.input_field.setEnabled(True)
        self.input_field.setFocus()

    def _add_bubble(self, text, is_user, model_name=None):
        bubble = ChatBubble(text, is_user, model_name)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, bubble)
        QTimer.singleShot(50, lambda: self._scroll.verticalScrollBar().setValue(
            self._scroll.verticalScrollBar().maximum()
        ))


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    pal = QPalette()
    pal.setColor(QPalette.Window,      QColor(9, 13, 22))
    pal.setColor(QPalette.WindowText,  QColor(241, 245, 249))
    pal.setColor(QPalette.Base,        QColor(15, 23, 42))
    pal.setColor(QPalette.Text,        QColor(241, 245, 249))
    pal.setColor(QPalette.Button,      QColor(124, 58, 237))
    pal.setColor(QPalette.ButtonText,  Qt.white)
    app.setPalette(pal)

    win = AIAssistant()
    win.show()
    sys.exit(app.exec_())
