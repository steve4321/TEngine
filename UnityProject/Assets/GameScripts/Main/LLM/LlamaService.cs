using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using UnityEngine;
using IntPtr = System.IntPtr;

namespace DigitalHuman.LLM
{
    /// <summary>
    /// 基于 llama.cpp 的本地 LLM 推理服务。
    /// 跨平台：Windows / macOS / Linux / Android / iOS。
    /// 在后台线程执行推理，通过回调返回结果到主线程。
    /// </summary>
    public class LlamaService : IDisposable
    {
        private IntPtr _model = IntPtr.Zero;
        private IntPtr _vocab = IntPtr.Zero;
        private IntPtr _context = IntPtr.Zero;
        private IntPtr _chatTemplate = IntPtr.Zero;
        private IntPtr _sampler = IntPtr.Zero;

        private bool _initialized;
        private readonly object _lock = new object();

        // 推理参数
        private int _nCtx;
        private int _nBatch = 512;
        private int _nPast;

        // 对话历史
        private readonly List<ChatMessage> _history = new List<ChatMessage>();

        public bool IsInitialized => _initialized;

        /// <summary>
        /// 初始化后端。程序启动时调用一次。
        /// </summary>
        public static void InitBackend()
        {
            // 加载所有 ggml 后端（CUDA/CPU/Metal 等）
            string pluginsPath = System.IO.Path.Combine(
                Application.dataPath, "Plugins", "Windows", "x86_64");
            IntPtr pathPtr = Marshal.StringToHGlobalAnsi(pluginsPath);
            try
            {
                Debug.Log("[LlamaService] Loading ggml backends from: " + pluginsPath);
                Native.ggml_backend_load_all_from_path(pathPtr);
                Debug.Log("[LlamaService] All ggml backends loaded - OK");
            }
            finally
            {
                Marshal.FreeHGlobal(pathPtr);
            }

            Native.llama_backend_init();
            Debug.Log("[LlamaService] Backend initialized");
        }

        /// <summary>
        /// 释放后端。程序退出时调用。
        /// </summary>
        public static void FreeBackend()
        {
            Native.llama_backend_free();
        }

        /// <summary>
        /// 加载 GGUF 模型并创建推理上下文。
        /// 耗时操作，应在后台线程调用。
        /// </summary>
        /// <param name="modelPath">GGUF 模型文件绝对路径</param>
        /// <param name="contextSize">上下文窗口大小</param>
        /// <param name="threads">推理线程数（0 = 自动）</param>
        /// <param name="systemPrompt">系统提示词（可选）</param>
        public void LoadModel(string modelPath, uint contextSize = 4096, int threads = 0, string systemPrompt = null)
        {
            lock (_lock)
            {
                if (_initialized) throw new InvalidOperationException("Model already loaded. Call Dispose first.");

                Debug.Log($"[LlamaService] Loading model: {modelPath}");

                // 1. 加载模型
                var modelParams = Native.llama_model_default_params();
                modelParams.nGpuLayers = 99;  // 全部层 GPU 加速
                modelParams.useMmap = false;
                modelParams.useMlock = false;

                Debug.Log($"[LlamaService] Model params: nGpuLayers={modelParams.nGpuLayers}, useMmap={modelParams.useMmap}, useMlock={modelParams.useMlock}");

                // 尝试用 model_desc 检查模型是否能被识别（vocab_only 模式）
                var vocabParams = Native.llama_model_default_params();
                vocabParams.nGpuLayers = 0;
                vocabParams.vocabOnly = true;
                IntPtr vocabModel = Native.llama_model_load_from_file(modelPath, vocabParams);
                if (vocabModel == IntPtr.Zero)
                {
                    Debug.LogError("[LlamaService] Vocab-only load also failed. The DLL cannot parse this GGUF file.");
                    Debug.LogError("[LlamaService] Possible cause: DLL version does not match GGUF format, or dependency DLL is missing.");
                    throw new Exception($"Failed to load model (vocab-only test also failed): {modelPath}");
                }
                else
                {
                    // vocab-only 成功，说明文件格式没问题
                    var descBuf = new StringBuilder(256);
                    Native.llama_model_desc(vocabModel, descBuf, 256);
                    Debug.Log($"[LlamaService] Model vocab-only OK: {descBuf}");
                    Native.llama_model_free(vocabModel);
                }

                _model = Native.llama_model_load_from_file(modelPath, modelParams);
                if (_model == IntPtr.Zero)
                {
                    Debug.LogError($"[LlamaService] Vocab-only worked but full load failed - likely OOM or tensor decode issue.");
                    throw new Exception($"Failed to load model (full load): {modelPath}");
                }

                // 2. 获取 vocab
                _vocab = Native.llama_model_get_vocab(_model);
                if (_vocab == IntPtr.Zero)
                    throw new Exception("Failed to get vocab from model");

                // 3. 创建上下文
                var ctxParams = Native.llama_context_default_params();
                ctxParams.nCtx = contextSize;
                ctxParams.nBatch = (uint)_nBatch;
                ctxParams.nUbatch = (uint)_nBatch;
                ctxParams.nSeqMax = 1;
                ctxParams.nThreads = threads;  // 已在主线程解析，> 0
                ctxParams.nThreadsBatch = ctxParams.nThreads;

                _context = Native.llama_init_from_model(_model, ctxParams);
                if (_context == IntPtr.Zero)
                    throw new Exception("Failed to create context");

                _nCtx = (int)Native.llama_n_ctx(_context);

                // 4. Warmup
                Warmup();

                // 5. 创建 sampler
                var samplerChainParams = Native.llama_sampler_chain_default_params();
                _sampler = Native.llama_sampler_chain_init(samplerChainParams);
                Native.llama_sampler_chain_add(_sampler, Native.llama_sampler_init_top_k(40));
                Native.llama_sampler_chain_add(_sampler, Native.llama_sampler_init_top_p(0.95f, 1));
                Native.llama_sampler_chain_add(_sampler, Native.llama_sampler_init_min_p(0.05f, 1));
                Native.llama_sampler_chain_add(_sampler, Native.llama_sampler_init_temp(0.8f));
                Native.llama_sampler_chain_add(_sampler, Native.llama_sampler_init_dist((uint)DateTime.Now.Ticks));

                // 6. 获取 chat template
                _chatTemplate = Native.llama_model_chat_template(_model, null);

                // 7. 处理 system prompt
                if (!string.IsNullOrEmpty(systemPrompt))
                {
                    _history.Add(new ChatMessage { Role = "system", Content = systemPrompt });
                    string formatted = FormatMessage(systemPrompt, "system", false);
                    var tokens = Tokenize(formatted, true, true);
                    FeedTokens(tokens);
                }

                _initialized = true;
                Debug.Log($"[LlamaService] Model loaded. Context: {_nCtx}, Threads: {ctxParams.nThreads}");
            }
        }

        /// <summary>
        /// 发送用户消息并获取回复。
        /// 在后台线程执行，逐 token 通过 onToken 回调返回。
        /// onToken 回调在后台线程执行，UI 更新需切回主线程。
        /// </summary>
        public string Chat(string userMessage, Action<string> onToken = null, CancellationToken ct = default)
        {
            if (!_initialized) throw new InvalidOperationException("Model not loaded");

            lock (_lock)
            {
                // 1. 格式化并 tokenize 用户消息
                _history.Add(new ChatMessage { Role = "user", Content = userMessage });
                string formatted = FormatMessage(userMessage, "user", true);
                var inputTokens = Tokenize(formatted, false, true);

                // 2. Feed input tokens
                FeedTokens(inputTokens);

                // 3. 生成回复
                var response = new StringBuilder();
                int nDecode = 0;

                while (!ct.IsCancellationRequested)
                {
                    // Sample next token
                    int newToken = Native.llama_sampler_sample(_sampler, _context, -1);

                    // Check end of generation
                    if (Native.llama_vocab_is_eog(_vocab, newToken))
                        break;

                    // Convert token to text
                    string piece = TokenToPiece(newToken);
                    response.Append(piece);
                    onToken?.Invoke(piece);

                    // Feed the token back for next iteration
                    var tokenArr = new int[] { newToken };
                    int result = DecodeBatch(tokenArr, 1);
                    if (result != 0)
                        throw new Exception($"Decode error: {result}");

                    Native.llama_sampler_accept(_sampler, newToken);
                    _nPast++;
                    nDecode++;

                    // Safety: limit max tokens per response
                    if (nDecode > 2048)
                        break;
                }

                string responseStr = response.ToString();
                _history.Add(new ChatMessage { Role = "assistant", Content = responseStr });

                // Reset sampler for next turn
                Native.llama_sampler_reset(_sampler);

                return responseStr;
            }
        }

        public void Dispose()
        {
            lock (_lock)
            {
                if (_sampler != IntPtr.Zero)
                {
                    Native.llama_sampler_free(_sampler);
                    _sampler = IntPtr.Zero;
                }

                if (_context != IntPtr.Zero)
                {
                    Native.llama_free(_context);
                    _context = IntPtr.Zero;
                }

                if (_model != IntPtr.Zero)
                {
                    Native.llama_model_free(_model);
                    _model = IntPtr.Zero;
                }

                _initialized = false;
                _nPast = 0;
                _history.Clear();
            }
        }

        #region Private Methods

        /// <summary>
        /// 解码 token batch，期间 pin 住 managed array 防止 GC 移动指针。
        /// </summary>
        private int DecodeBatch(int[] tokens, int nTokens)
        {
            GCHandle handle = GCHandle.Alloc(tokens, GCHandleType.Pinned);
            try
            {
                var batch = Native.llama_batch_get_one(tokens, nTokens);
                return Native.llama_decode(_context, batch);
            }
            finally
            {
                handle.Free();
            }
        }

        private void Warmup()
        {
            Native.llama_set_warmup(_context, true);

            var tmp = new List<int>();
            int bos = Native.llama_vocab_bos(_vocab);
            int eos = Native.llama_vocab_eos(_vocab);
            if (bos != -1) tmp.Add(bos);
            if (eos != -1) tmp.Add(eos);
            if (tmp.Count == 0) tmp.Add(0);

            if (Native.llama_model_has_decoder(_model))
            {
                var tmpArr = tmp.ToArray();
                DecodeBatch(tmpArr, tmp.Count);
            }

            Native.llama_memory_clear(Native.llama_get_memory(_context), true);
            Native.llama_synchronize(_context);
            Native.llama_perf_context_reset(_context);
            Native.llama_set_warmup(_context, false);
        }

        private int[] Tokenize(string text, bool addBos, bool special)
        {
            // 中文文本每个字符可能产生 2-4 个 token，预留足够空间
            int nMaxTokens = text.Length * 4 + 256;
            var tokens = new int[nMaxTokens];
            int nTokens = Native.llama_tokenize(_vocab, text, text.Length, tokens, nMaxTokens, addBos, special);
            if (nTokens < 0)
                throw new Exception($"Tokenization failed: {nTokens}");
            var result = new int[nTokens];
            Array.Copy(tokens, result, nTokens);
            return result;
        }

        private string TokenToPiece(int token)
        {
            var buf = new StringBuilder(64);
            int len = Native.llama_token_to_piece(_vocab, token, buf, buf.Capacity, 0, true);
            if (len < 0) return "";
            if (len >= buf.Capacity)
            {
                buf = new StringBuilder(len + 1);
                Native.llama_token_to_piece(_vocab, token, buf, buf.Capacity, 0, true);
            }
            return buf.ToString();
        }

        private void FeedTokens(int[] tokens)
        {
            for (int i = 0; i < tokens.Length; i += _nBatch)
            {
                int chunkLen = Math.Min(_nBatch, tokens.Length - i);
                var chunk = new int[chunkLen];
                Array.Copy(tokens, i, chunk, 0, chunkLen);

                int result = DecodeBatch(chunk, chunkLen);
                if (result != 0)
                    throw new Exception($"Decode error: {result}, past: {_nPast}, chunk: {chunkLen}");

                _nPast += chunkLen;
            }
            foreach (int t in tokens)
                Native.llama_sampler_accept(_sampler, t);
        }

        private string FormatMessage(string content, string role, bool isLast)
        {
            // 简单的 chat template 格式化
            // TODO: 使用 llama_model_chat_template 做更精确的格式化
            if (_chatTemplate != IntPtr.Zero)
            {
                // 使用 llama_chat_format_single（如果可用的话）
                // 暂时用简单格式
            }

            string prefix = role switch
            {
                "system" => "<|im_start|>system\n",
                "user" => "<|im_start|>user\n",
                "assistant" => "<|im_start|>assistant\n",
                _ => ""
            };

            string suffix = isLast ? "<|im_end|>\n<|im_start|>assistant\n" : "<|im_end|>\n";

            return $"{prefix}{content}\n{suffix}";
        }

        #endregion

        private struct ChatMessage
        {
            public string Role;
            public string Content;
        }
    }
}
