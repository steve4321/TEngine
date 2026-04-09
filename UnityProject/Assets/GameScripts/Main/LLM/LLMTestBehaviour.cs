using System;
using System.Threading;
using UnityEngine;
using UnityEngine.UI;

namespace DigitalHuman.LLM
{
    /// <summary>
    /// LLM 推理测试脚本。
    /// 挂到任意 GameObject 上，配置模型路径，运行后自动测试推理。
    /// </summary>
    public class LLMTestBehaviour : MonoBehaviour
    {
        [Header("模型配置")]
        [Tooltip("GGUF 模型绝对路径")]
        public string modelPath = "";

        [Tooltip("上下文窗口大小")]
        public uint contextSize = 2048;  // 降低减少 GPU 显存占用

        [Tooltip("推理线程数（0=自动）")]
        public int threads = 0;

        [Header("对话配置")]
        [TextArea(3, 5)]
        public string systemPrompt = "你是一个友好的数字人伙伴，用简短自然的中文回复。";

        [Header("测试")]
        [TextArea(2, 3)]
        public string testMessage = "你好，介绍一下你自己";

        [Header("UI（可选）")]
        public Text outputText;

        private LlamaService _service;
        private Thread _inferenceThread;
        private readonly object _uiLock = new object();
        private string _uiText = "";
        private int _cachedThreads;

        private void Start()
        {
            // 如果没有配置路径，尝试 StreamingAssets
            if (string.IsNullOrEmpty(modelPath))
            {
                modelPath = System.IO.Path.Combine(Application.streamingAssetsPath, "Models", "qwen3.5-0.8b-q4_k_m.gguf");
            }

            // SystemInfo 只能在主线程访问，提前缓存
            _cachedThreads = threads > 0 ? threads : SystemInfo.processorCount;

            Log($"模型路径: {modelPath}");
            Log($"平台: {Application.platform}");
            Log($"处理器: {SystemInfo.processorType}, 核心数: {SystemInfo.processorCount}, 使用线程: {_cachedThreads}");

            // 初始化后端（主线程）
            LlamaService.InitBackend();

            // 在后台线程加载模型
            _inferenceThread = new Thread(LoadAndTest);
            _inferenceThread.IsBackground = true;
            _inferenceThread.Start();
        }

        private void LoadAndTest()
        {
            try
            {
                _service = new LlamaService();

                Log("正在加载模型...");
                var sw = System.Diagnostics.Stopwatch.StartNew();

                _service.LoadModel(modelPath, contextSize, _cachedThreads, systemPrompt);

                sw.Stop();
                Log($"模型加载完成，耗时 {sw.ElapsedMilliseconds}ms");

                // 测试推理
                Log($"\n用户: {testMessage}");
                Log("AI 思考中...");

                sw.Restart();
                string response = _service.Chat(testMessage, onToken: (token) =>
                {
                    // 流式 token 回调在后台线程
                    // 简单拼接，主线程 Update 里刷新 UI
                    lock (_uiLock) { _uiText += token; }
                });
                sw.Stop();

                Log($"\nAI: {response}");
                Log($"推理完成，耗时 {sw.ElapsedMilliseconds}ms");
            }
            catch (Exception ex)
            {
                Log($"错误: {ex.Message}\n{ex.StackTrace}");
            }
        }

        private void Update()
        {
            // 刷新 UI（主线程）
            if (outputText != null)
            {
                lock (_uiLock)
                {
                    if (_uiText.Length > 0)
                    {
                        outputText.text += _uiText;
                        _uiText = "";
                    }
                }
            }
        }

        private void OnDestroy()
        {
            _service?.Dispose();
            LlamaService.FreeBackend();
        }

        private void Log(string msg)
        {
            Debug.Log($"[LLMTest] {msg}");
            lock (_uiLock) { _uiText += msg + "\n"; }
        }
    }
}
