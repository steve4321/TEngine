using System;
using System.Runtime.InteropServices;
using System.Text;

namespace DigitalHuman.LLM
{
    /// <summary>
    /// llama.cpp C API 的 P/Invoke 封装。
    /// 只暴露对话推理所需的最小 API 子集。
    /// 原生库名：Windows = llama.dll, macOS/Linux = libllama.so / libllama.dylib
    /// </summary>
    internal static class Native
    {
        // Unity 会根据平台自动加载正确的原生库
        // Windows: Assets/Plugins/Windows/x86_64/llama.dll
        // macOS:   Assets/Plugins/macOS/libllama.dylib
        // Linux:   Assets/Plugins/Linux/x86_64/libllama.so
        // Android: Assets/Plugins/Android/arm64-v8a/libllama.so
        // iOS:     XCFramework
        private const string DLL_NAME = "llama";

        #region Backend

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_backend_init();

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_backend_free();

        // ggml_backend_load 在 ggml.dll 中
        [DllImport("ggml", CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr ggml_backend_load(IntPtr path);

        [DllImport("ggml", CallingConvention = CallingConvention.Cdecl)]
        public static extern void ggml_backend_load_all();

        [DllImport("ggml", CallingConvention = CallingConvention.Cdecl)]
        public static extern void ggml_backend_load_all_from_path(IntPtr path);

        #endregion

        #region Model

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern LLamaModelParams llama_model_default_params();

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_model_load_from_file(
            string pathModel,
            LLamaModelParams parameters);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_model_free(IntPtr model);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_model_desc(IntPtr model, StringBuilder buf, int bufSize);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_model_get_vocab(IntPtr model);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_model_chat_template(IntPtr model, string name);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_model_n_ctx_train(IntPtr model);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_model_n_embd(IntPtr model);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern bool llama_model_has_encoder(IntPtr model);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern bool llama_model_has_decoder(IntPtr model);

        #endregion

        #region Context

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern LLamaContextParams llama_context_default_params();

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_init_from_model(
            IntPtr model,
            LLamaContextParams parameters);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_free(IntPtr ctx);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern uint llama_n_ctx(IntPtr ctx);

        #endregion

        #region Vocab / Tokenization

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_vocab_n_tokens(IntPtr vocab);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_vocab_bos(IntPtr vocab);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_vocab_eos(IntPtr vocab);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern bool llama_vocab_is_eog(IntPtr vocab, int token);

        /// <summary>
        /// 将文本 tokenize 为 token ID 数组。
        /// 返回实际写入的 token 数量。
        /// </summary>
        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_tokenize(
            IntPtr vocab,
            string text,
            int textLen,
            [Out] int[] tokens,
            int nMaxTokens,
            bool addBos,
            bool special);

        /// <summary>
        /// 将单个 token 转为文本字符串。
        /// 返回写入 buf 的字节数（不含 null），负数表示错误。
        /// </summary>
        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_token_to_piece(
            IntPtr vocab,
            int token,
            StringBuilder buf,
            int length,
            int lstrip,
            bool special);

        #endregion

        #region Batch / Decode

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern LLamaBatch llama_batch_get_one(
            [In] int[] tokens,
            int nTokens);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_decode(IntPtr ctx, LLamaBatch batch);

        #endregion

        #region Memory

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_get_memory(IntPtr ctx);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_memory_clear(IntPtr mem, bool data);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern bool llama_memory_seq_rm(
            IntPtr mem, int seqId, int p0, int p1);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_memory_seq_add(
            IntPtr mem, int seqId, int p0, int p1, int delta);

        #endregion

        #region Sampler (新版 API)

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern LLamaSamplerChainParams llama_sampler_chain_default_params();

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_sampler_chain_init(LLamaSamplerChainParams @params);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_sampler_free(IntPtr sampler);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_sampler_init_temp(float temp);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_sampler_init_dist(uint seed);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_sampler_init_top_k(int k);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_sampler_init_top_p(float p, int minKeep);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_sampler_init_min_p(float p, int minKeep);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_sampler_init_repeat_penalty(
            int nPrev, int penaltyLastN,
            float penaltyRepeat, float penaltyFreq, float penaltyPresent);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_sampler_chain_add(IntPtr chain, IntPtr sampler);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_sampler_sample(IntPtr smpl, IntPtr ctx, int idx);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_sampler_accept(IntPtr smpl, int token);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_sampler_reset(IntPtr smpl);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_sampler_last(IntPtr smpl);

        #endregion

        #region Misc

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_synchronize(IntPtr ctx);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_perf_context_reset(IntPtr ctx);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_set_warmup(IntPtr ctx, bool warmup);

        #endregion

        #region Structs

        [StructLayout(LayoutKind.Sequential)]
        public struct LLamaModelParams
        {
            public IntPtr devices;
            public IntPtr tensorBuftOverrides;
            public int nGpuLayers;
            public int splitMode;
            public int mainGpu;
            public IntPtr tensorSplit;
            public IntPtr progressCallback;
            public IntPtr progressCallbackUserData;
            public IntPtr kvOverrides;
            // bool fields
            [MarshalAs(UnmanagedType.U1)] public bool vocabOnly;
            [MarshalAs(UnmanagedType.U1)] public bool useMmap;
            [MarshalAs(UnmanagedType.U1)] public bool useDirectIo;
            [MarshalAs(UnmanagedType.U1)] public bool useMlock;
            [MarshalAs(UnmanagedType.U1)] public bool checkTensors;
            [MarshalAs(UnmanagedType.U1)] public bool useExtraBufts;
            [MarshalAs(UnmanagedType.U1)] public bool noHost;
            [MarshalAs(UnmanagedType.U1)] public bool noAlloc;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct LLamaContextParams
        {
            public uint nCtx;
            public uint nBatch;
            public uint nUbatch;
            public uint nSeqMax;
            public int nThreads;
            public int nThreadsBatch;
            // rope/pooling/attention/flash params...
            public int ropeScalingType;
            public int poolingType;
            public int attentionType;
            public int flashAttnType;
            public float ropeFreqBase;
            public float ropeFreqScale;
            public float yarnExtFactor;
            public float yarnAttnFactor;
            public float yarnBetaFast;
            public float yarnBetaSlow;
            public uint yarnOrigCtx;
            public float defragThold;
            // callbacks
            public IntPtr cbEval;
            public IntPtr cbEvalUserData;
            // cache types
            public int typeK;
            public int typeV;
            // abort callback
            public IntPtr abortCallback;
            public IntPtr abortCallbackData;
            // bool fields
            [MarshalAs(UnmanagedType.U1)] public bool embeddings;
            [MarshalAs(UnmanagedType.U1)] public bool offloadKqv;
            [MarshalAs(UnmanagedType.U1)] public bool noPerf;
            [MarshalAs(UnmanagedType.U1)] public bool opOffload;
            [MarshalAs(UnmanagedType.U1)] public bool swaFull;
            [MarshalAs(UnmanagedType.U1)] public bool kvUnified;
            // sampler config
            public IntPtr samplers;
            public ulong nSamplers;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct LLamaBatch
        {
            public int nTokens;
            public IntPtr token;
            public IntPtr embd;
            public IntPtr pos;
            public IntPtr nSeqId;
            public IntPtr seqId;
            public IntPtr logits;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct LLamaSamplerChainParams
        {
            [MarshalAs(UnmanagedType.U1)] public bool noPerf;
        }

        #endregion
    }
}
