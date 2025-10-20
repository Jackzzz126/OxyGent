"""
流程图生成工具模块
"""

from oxygent.oxy import FunctionHub
import os
import json
import requests
import webbrowser
from pathlib import Path
import time
import datetime
from dotenv import load_dotenv

# 初始化 FunctionHub
flow_image_gen_tools = FunctionHub(name="flow_image_gen_tools")

# API 配置 - 使用 OpenAI 兼容接口
API_BASE_URL = "http://llm-32b.jd.com/v1"
API_KEY = "EMPTY"
MODEL_NAME = "qwen25-32b-native"

# 默认提示词模板
DEFAULT_PROMPT_TEMPLATE = """
请根据以下描述生成一个 Mermaid 流程图代码：

{description}

不需要询问更多信息，直接根据描述生成一个完整的流程图。
流程图应该包括所有提到的阶段，并且使用合适的图形元素和连接线。

请只返回 Mermaid 代码，不要包含任何其他解释、说明或问题。代码应该以 ```mermaid 开头，以 ``` 结尾。

示例格式：
```mermaid
flowchart TD
    A[开始] --> B[步骤1]
    B --> C[步骤2]
    C --> D[结束]
```
"""

@flow_image_gen_tools.tool(
    description="根据文本描述生成 Mermaid 流程图并返回 HTML 文件路径。此工具使用 Ollama API 将文本描述转换为 Mermaid 流程图代码，然后生成可视化的 HTML 文件并在浏览器中打开。"
)
async def generate_flow_chart(description: str, output_path: str = None) -> str:
    """
    根据文本描述生成 Mermaid 流程图并在浏览器中打开
    
    Args:
        description: 流程图的文本描述
        output_path: 输出的 HTML 文件路径，默认为 "flowchart.html"
        
    Returns:
        str: 生成的 HTML 文件的路径
    """
    try:
        # 如果没有提供输出路径，则生成带有时间戳的默认文件名
        if output_path is None:
            # 使用绝对路径，确保文件创建在正确的位置
            output_path = os.path.abspath(f"output/software-development-workflow.html")
        else:
            # 确保输出路径是绝对路径
            output_path = os.path.abspath(output_path)
            
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
        
        # 调用 OpenAI 兼容 API 生成 Mermaid 代码
        mermaid_code = call_openai_api(description)
        
        # 创建 HTML 文件并渲染流程图
        if create_html_with_mermaid(mermaid_code, output_path):
            # 自动在浏览器中打开生成的文件
            import webbrowser
            try:
                webbrowser.open(f"file://{output_path}")
                return f"✅ 流程图已生成并在浏览器中打开: {output_path}"
            except Exception as e:
                print(f"打开浏览器时出错: {e}")
                return f"✅ 流程图已生成并保存到: {output_path}，请手动打开文件"
        else:
            return "❌ 生成流程图时出错"
    except Exception as e:
        print(f"generate_flow_chart 函数执行出错: {e}")
        return f"❌ 生成流程图时出错: {str(e)}"

def call_openai_api(description):
    """调用 OpenAI 兼容 API 生成 Mermaid 代码"""
    try:
        prompt = DEFAULT_PROMPT_TEMPLATE.format(description=description)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
        
        # OpenAI API 格式
        data = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 2000,
            "stream": False
        }
        
        print("正在调用 OpenAI 兼容 API 生成流程图代码...")
        print(f"请求URL: {API_BASE_URL}/chat/completions")
        print(f"请求模型: {MODEL_NAME}")
        
        response = requests.post(f"{API_BASE_URL}/chat/completions", headers=headers, json=data, timeout=30)
        
        print(f"API 响应状态: {response.status_code}")
        
        if response.status_code != 200:
            print(f"API 请求失败，状态码: {response.status_code}")
            print(f"响应内容: {response.text}")
            return generate_sample_mermaid()
        
        result = response.json()
        
        # OpenAI API 标准格式
        if "choices" in result and len(result.get("choices", [])) > 0:
            content = result["choices"][0]["message"]["content"]
            print(f"API 调用成功，内容长度: {len(content)}")
        else:
            print(f"无法识别的 API 响应格式: {result}")
            return generate_sample_mermaid()
            
        # 提取 Mermaid 代码
        mermaid_code = extract_mermaid_code(content)
        if not mermaid_code:
            print("未能从 API 响应中提取有效的 Mermaid 代码，将使用示例流程图")
            return generate_sample_mermaid()
        
        print("成功提取 Mermaid 代码")
        return mermaid_code
        
    except requests.exceptions.RequestException as e:
        print(f"网络请求错误: {e}")
        return generate_sample_mermaid()
    except Exception as e:
        print(f"调用 API 时出错: {e}")
        return generate_sample_mermaid()

def extract_mermaid_code(content):
    """从 API 响应中提取 Mermaid 代码"""
    # 尝试提取 ```mermaid ... ``` 格式的代码块
    if "```mermaid" in content and "```" in content.split("```mermaid", 1)[1]:
        return content.split("```mermaid", 1)[1].split("```", 1)[0].strip()
    # 如果没有明确的标记，尝试提取看起来像 Mermaid 代码的部分
    elif any(keyword in content.lower() for keyword in ["graph ", "flowchart ", "sequencediagram", "classDiagram"]):
        lines = content.split('\n')
        start_idx = None
        end_idx = None
        
        for i, line in enumerate(lines):
            if start_idx is None and any(keyword in line.lower() for keyword in ["graph ", "flowchart ", "sequencediagram", "classDiagram"]):
                start_idx = i
            elif start_idx is not None and line.strip() == "" and i > start_idx + 3:
                end_idx = i
                break
        
        if start_idx is not None:
            end_idx = end_idx or len(lines)
            return '\n'.join(lines[start_idx:end_idx]).strip()
    
    return None

def generate_sample_mermaid():
    """生成示例 Mermaid 流程图代码"""
    return """flowchart TD
    A[需求分析] --> B[系统设计]
    B --> C[技术选型]
    C --> D[架构设计]
    D --> E[编码实现]
    E --> F[单元测试]
    F --> G[集成测试]
    G --> H{测试通过?}
    H -->|是| I[代码审查]
    H -->|否| E
    I --> J[部署准备]
    J --> K[生产部署]
    K --> L[监控运维]
    L --> M[用户反馈]
    M --> N{需要优化?}
    N -->|是| A
    N -->|否| O[项目完成]
    
    style A fill:#e1f5fe
    style O fill:#c8e6c9
    style H fill:#fff3e0
    style N fill:#fff3e0"""

def create_html_with_mermaid(mermaid_code, output_path):
    """创建包含可交互编辑的 Mermaid 流程图的 HTML 文件"""
    html_content = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>交互式 Mermaid 流程图编辑器</title>
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.2/codemirror.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.2/mode/javascript/javascript.min.js"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.2/codemirror.min.css">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.2/theme/dracula.min.css">
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            h1, h2 {{
                color: #333;
                text-align: center;
            }}
            .editor-container {{
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                margin: 20px 0;
            }}
            .editor-panel {{
                flex: 1;
                min-width: 300px;
            }}
            .preview-panel {{
                flex: 1;
                min-width: 300px;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 10px;
                background-color: white;
            }}
            .CodeMirror {{
                height: 400px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }}
            .button-group {{
                margin: 20px 0;
                text-align: center;
            }}
            button {{
                background-color: #4CAF50;
                color: white;
                padding: 10px 15px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                margin: 0 5px;
                font-size: 14px;
            }}
            button:hover {{
                background-color: #45a049;
            }}
            .template-selector {{
                margin: 20px 0;
                text-align: center;
            }}
            select {{
                padding: 8px;
                border-radius: 4px;
                border: 1px solid #ddd;
                font-size: 14px;
                min-width: 200px;
            }}
            .mermaid {{
                display: flex;
                justify-content: center;
                margin: 20px 0;
                min-height: 200px;
                border: 1px solid #eee;
                padding: 10px;
                border-radius: 4px;
                background-color: #fafafa;
            }}
            .mermaid svg {{
                max-width: 100%;
                height: auto !important;
            }}
            .error-message {{
                color: #d32f2f;
                background-color: #ffebee;
                padding: 10px;
                border-radius: 4px;
                margin-top: 10px;
                border-left: 4px solid #d32f2f;
                font-family: monospace;
            }}
            .footer {{
                margin-top: 30px;
                text-align: center;
                color: #666;
                font-size: 0.9em;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>交互式 Mermaid 流程图编辑器</h1>
            
            <div class="template-selector">
                <label for="template-select">选择模板：</label>
                <select id="template-select" onchange="loadTemplate()">
                    <option value="custom">自定义</option>
                    <option value="software-dev">软件开发流程</option>
                    <option value="project-management">项目管理流程</option>
                    <option value="business-process">业务流程</option>
                    <option value="decision-tree">决策树</option>
                </select>
            </div>
            
            <div class="editor-container">
                <div class="editor-panel">
                    <h2>Mermaid 代码编辑器</h2>
                    <textarea id="code-editor">{mermaid_code}</textarea>
                </div>
                
                <div class="preview-panel">
                    <h2>实时预览</h2>
                    <div id="preview" class="mermaid">
{mermaid_code}
                    </div>
                    <div id="render-error" class="error-message" style="display:none;"></div>
                </div>
            </div>
            
            <div class="button-group">
                <button onclick="updatePreview()">更新预览</button>
                <button onclick="exportSVG()">导出 SVG</button>
                <button onclick="exportPNG()">导出 PNG</button>
            </div>
            
            <div class="footer">
                <p>由 OxyGent 和 Mermaid.js 提供支持 | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        </div>
        
        <script>
            // 初始化 Mermaid
            mermaid.initialize({{
                startOnLoad: true,
                theme: 'default',
                securityLevel: 'loose',
                flowchart: {{
                    htmlLabels: true,
                    curve: 'basis'
                }},
                er: {{
                    useMaxWidth: false
                }},
                gantt: {{
                    useMaxWidth: false
                }}
            }});
            
            // 初始化代码编辑器
            var editor = CodeMirror.fromTextArea(document.getElementById("code-editor"), {{
                mode: "javascript",
                theme: "dracula",
                lineNumbers: true,
                autoCloseBrackets: true,
                matchBrackets: true,
                tabSize: 2,
                indentWithTabs: true
            }});
            
            // 模板库
            const templates = {{
                "software-dev": `flowchart TD
    A[需求分析] --> B[系统设计]
    B --> C[编码实现]
    C --> D[测试]
    D --> E{"测试通过?"}
    E -->|是| F[部署]
    E -->|否| C
    F --> G[维护]`,
                "project-management": `flowchart TD
    A[项目启动] --> B[需求收集]
    B --> C[项目规划]
    C --> D[执行]
    D --> E[监控与控制]
    E --> F{"需要变更?"}
    F -->|是| G[变更控制]
    G --> D
    F -->|否| H[项目收尾]`,
                "business-process": `flowchart TD
    A[开始] --> B[接收订单]
    B --> C[库存检查]
    C --> D{"库存充足?"}
    D -->|是| E[处理付款]
    D -->|否| F[通知缺货]
    F --> G[补充库存]
    G --> E
    E --> H[发货]
    H --> I[结束]`,
                "decision-tree": `flowchart TD
    A[问题] --> B{"条件1?"}
    B -->|是| C[结果1]
    B -->|否| D{"条件2?"}
    D -->|是| E[结果2]
    D -->|否| F[结果3]`
            }};
            
            // 加载模板
            function loadTemplate() {{
                const select = document.getElementById("template-select");
                const templateName = select.value;
                
                if (templateName !== "custom") {{
                    editor.setValue(templates[templateName]);
                    updatePreview();
                }}
            }}
            
            // 更新预览
            function updatePreview() {{
                const code = editor.getValue();
                const previewDiv = document.getElementById("preview");
                const errorDiv = document.getElementById("render-error");
                
                try {{
                    // 完全重新创建预览区域，避免渲染问题
                    const previewContainer = previewDiv.parentNode;
                    const oldPreview = previewDiv;
                    
                    // 创建新的预览div
                    const newPreview = document.createElement("div");
                    newPreview.id = "preview";
                    newPreview.className = "mermaid";
                    newPreview.textContent = code;
                    
                    // 替换旧的预览div
                    previewContainer.replaceChild(newPreview, oldPreview);
                    
                    // 重新初始化Mermaid
                    mermaid.initialize({{
                        startOnLoad: false,  // 不自动启动
                        theme: 'default',
                        securityLevel: 'loose',
                        flowchart: {{
                            htmlLabels: true,
                            curve: 'basis'
                        }}
                    }});
                    
                    // 手动渲染
                    setTimeout(() => {{
                        try {{
                            mermaid.init(undefined, "#preview");
                            // 隐藏错误信息
                            errorDiv.style.display = "none";
                        }} catch (innerError) {{
                            console.error("Mermaid渲染错误:", innerError);
                            errorDiv.textContent = "图表渲染错误: " + innerError.message;
                            errorDiv.style.display = "block";
                        }}
                    }}, 300);
                }} catch (e) {{
                    console.error("Mermaid渲染错误:", e);
                    errorDiv.textContent = "图表渲染错误: " + e.message;
                    errorDiv.style.display = "block";
                }}
            }}
            
            // 保存更改
            function saveChanges() {{
                const code = editor.getValue();
                
                // 发送到服务器保存
                fetch('/api/save-flowchart', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json',
                    }},
                    body: JSON.stringify({{ mermaid_code: code }}),
                }})
                .then(response => response.json())
                .then(data => {{
                    if (data.success) {{
                        alert('流程图已保存！文件路径: ' + data.file_path);
                    }} else {{
                        alert('保存失败: ' + data.error);
                    }}
                }})
                .catch(error => {{
                    console.error('保存出错:', error);
                    alert('保存出错，请查看控制台获取详细信息');
                }});
            }}
            
            // 导出 SVG
            function exportSVG() {{
                const svgCode = document.querySelector(".mermaid svg");
                if (svgCode) {{
                    const svgData = new XMLSerializer().serializeToString(svgCode);
                    const svgBlob = new Blob([svgData], {{type: 'image/svg+xml;charset=utf-8'}});
                    const svgUrl = URL.createObjectURL(svgBlob);
                    
                    const downloadLink = document.createElement("a");
                    downloadLink.href = svgUrl;
                    downloadLink.download = "flowchart.svg";
                    document.body.appendChild(downloadLink);
                    downloadLink.click();
                    document.body.removeChild(downloadLink);
                }} else {{
                    alert("无法导出 SVG，请先更新预览");
                }}
            }}
            
            // 导出 PNG
            function exportPNG() {{
                const svgElement = document.querySelector(".mermaid svg");
                if (svgElement) {{
                    const canvas = document.createElement("canvas");
                    const ctx = canvas.getContext("2d");
                    
                    // 创建图像
                    const svgData = new XMLSerializer().serializeToString(svgElement);
                    const img = new Image();
                    
                    img.onload = function() {{
                        canvas.width = img.width;
                        canvas.height = img.height;
                        ctx.drawImage(img, 0, 0);
                        
                        const pngUrl = canvas.toDataURL("image/png");
                        
                        const downloadLink = document.createElement("a");
                        downloadLink.href = pngUrl;
                        downloadLink.download = "flowchart.png";
                        document.body.appendChild(downloadLink);
                        downloadLink.click();
                        document.body.removeChild(downloadLink);
                    }};
                    
                    img.src = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(svgData)));
                }} else {{
                    alert("无法导出 PNG，请先更新预览");
                }}
            }}
            
            // 页面加载完成后初始化预览
            document.addEventListener('DOMContentLoaded', function() {{
                // 延迟一下以确保所有组件已加载
                setTimeout(function() {{
                    // 确保编辑器内容已加载
                    if (editor.getValue().trim() === '') {{
                        // 如果编辑器为空，尝试从预览区域获取内容
                        const previewContent = document.getElementById("preview").textContent.trim();
                        if (previewContent) {{
                            editor.setValue(previewContent);
                        }}
                    }}
                    
                    // 确保mermaid已完全加载
                    if (typeof mermaid !== 'undefined') {{
                        // 配置mermaid
                        mermaid.initialize({{
                            startOnLoad: false,
                            theme: 'default',
                            securityLevel: 'loose',
                            flowchart: {{
                                htmlLabels: true,
                                curve: 'basis'
                            }}
                        }});
                        
                        // 完全重新创建预览区域，避免渲染问题
                        const previewDiv = document.getElementById("preview");
                        const previewContainer = previewDiv.parentNode;
                        const code = editor.getValue();
                        
                        // 创建新的预览div
                        const newPreview = document.createElement("div");
                        newPreview.id = "preview";
                        newPreview.className = "mermaid";
                        newPreview.textContent = code;
                        
                        // 替换旧的预览div
                        previewContainer.replaceChild(newPreview, previewDiv);
                        
                        // 手动渲染
                        mermaid.init(undefined, "#preview");
                    }} else {{
                        console.error("Mermaid库未加载");
                        document.getElementById("render-error").textContent = "图表库未加载，请刷新页面";
                        document.getElementById("render-error").style.display = "block";
                    }}
                }}, 500);  // 增加延迟时间，确保所有资源加载完成
            }});
        </script>
    </body>
    </html>
    """
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"已保存流程图到: {output_path}")
        return True
    except Exception as e:
        print(f"保存 HTML 文件时出错: {e}")
        return False


if __name__ == "__main__":
    asyncio.run(main())