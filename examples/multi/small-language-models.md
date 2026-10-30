# State of Small Language Models for On‑Device Use (2026)

**Summary**  
Small transformer‑based language models such as TinyLlama‑1.1B are being positioned for on‑device inference, but publicly available evidence is limited to basic architectural details and a nascent software stack. Quantitative performance numbers, systematic latency measurements, and comparative analyses of dense versus sparsified or quantized variants are absent. The current deployment ecosystem centers on TensorFlow Lite and MediaPipe, with Android 14’s AICore offering a production‑grade path for Gemini‑based solutions. Significant gaps remain in hardware coverage, benchmark data, and advanced model‑compression techniques.

## Model Architectures and Size Trade‑offs  
TinyLlama‑1.1B contains approximately 1.1 billion parameters and follows the same transformer architecture and tokenizer as Llama 2, enabling drop‑in compatibility with Llama‑based projects [1]. The model was pretrained on a 3 trillion‑token corpus, with training commencing on 2023‑09‑01 using 16 A100‑40 GB GPUs [1]. It is marketed as a compact model suitable for applications with restricted computation and memory footprints, targeting on‑device deployment scenarios [1].

*Critic assessment*: This section was judged weak because the claim that the released model files are stored in 32‑bit floating‑point (F32) format is not supported by the provided excerpt. Four issues were identified, leaving four claims unverified. Moreover, the findings do not include quantitative performance metrics (e.g., perplexity, GLUE scores), latency or memory consumption for dense, sparsified, or quantized variants, nor do they discuss pruning, knowledge distillation, mixture‑of‑experts, or detailed 8‑bit/4‑bit/binary quantization trade‑offs. No comparison with other ≤10 B‑parameter models (e.g., Phi‑2, MobileBERT‑LLM) is provided.

## Software Stacks and Deployment Pipelines  
TensorFlow Lite combined with MediaPipe enables on‑device LLM inference across Web, Android, and iOS platforms [2]. The MediaPipe LLM Inference API supplies a conversion step that transforms model weights into a TensorFlow Lite flatbuffer via a MediaPipe Python package [2]. Currently the API supports four openly available LLMs—Gemma, Phi 2, Falcon, and Stable LM [2]. On‑device stack optimizations include new operators, quantization, caching, and weight‑sharing techniques to fit LLMs within limited memory and compute budgets [2].

For production Android deployments, developers are directed to use the Gemini API or Gemini Nano via Android AICore rather than the experimental MediaPipe API [2]. Android AICore, introduced in Android 14, provides system‑level integration with ML accelerators, LoRA adapters, and safety filters for Gemini‑powered solutions [2]. The typical workflow consists of selecting compatible model weights, converting them to a TensorFlow Lite flatbuffer, and integrating the resulting artifact with platform‑specific SDKs (Web, Android, iOS) [2]. The MediaPipe LLM Inference API is explicitly labeled for experimental and research use only on Android devices [2].

*Open questions*: Measured latencies for a 500 M‑parameter model on major mobile inference runtimes (TensorFlow Lite, ONNX Runtime Mobile, PyTorch Mobile, Apple Core ML, Qualcomm Snapdragon NPU SDK) are not provided. Information on operator fusion, dynamic batching, kernel selection, ease‑of‑integration assessments, OTA update mechanisms, security signing processes, and additional caching or weight‑sharing strategies beyond those described for MediaPipe is also missing.

## Missing or Weak Sections  
- **Hardware Landscape for On‑Device LLMs** – no usable findings were supplied.  
- **Use‑Case Landscape and Performance Benchmarks** – no usable findings were supplied.  
- **Challenges, Open Problems, and Future Directions** – no usable findings were supplied.  

## Gaps Identified  
- No comparison between dense, sparsified, and quantized transformer variants.  
- Absence of discussion of models ≤10 B parameters beyond TinyLlama (e.g., Phi‑2, MobileBERT‑LLM).  
- Lack of performance metrics such as latency, perplexity, or downstream task scores for the architectures considered.  
- No coverage of weight pruning, mixture‑of‑experts, or detailed 4‑bit/binary quantization techniques.  
- Impact of architecture choices on memory usage and on‑device latency is not quantified.

## Limits  
This report relies exclusively on the two supplied sources. Where the source material does not contain quantitative data, benchmark results, or detailed methodological descriptions, the report explicitly notes the absence rather than inferring or fabricating information. Consequently, the analysis is constrained to high‑level architectural and software‑stack observations, and many critical dimensions of on‑device LLM viability remain undocumented.

## Coverage notes

- Missing: "Hardware Landscape for On‑Device LLMs" has no usable findings (time limit reached).
- Weak: "Model Architectures and Size Trade‑offs" was judged weak by the critic (The claim that the released model files are stored in 32‑bit floating‑point (F32) format is not supported by the provided excerpt).
- Critic issues: "Model Architectures and Size Trade‑offs" has 4 issue(s); 4 claim(s) left out as unverified.
- Unverified, not used in the text: The released model files are stored in 32‑bit floating‑point (F32) format and are listed as a 1 B‑parameter model
- Unverified, not used in the text: The repository provides at least one quantized variant of TinyLlama (noted as “Quantizations: 1 model”)
- Unverified, not used in the text: The Hugging Face inference pipeline demonstrates loading TinyLlama in half‑precision (torch.float16), indicating support for 16‑bit inference on compatible hardware
- Unverified, not used in the text: TinyLlama’s model page recorded 585 downloads in the most recent month, reflecting moderate community interest
- Missing: "Use‑Case Landscape and Performance Benchmarks" has no usable findings (model output unusable).
- Missing: "Challenges, Open Problems, and Future Directions" has no usable findings (model output unusable).
- Gap: No comparison between dense, sparsified, and quantized transformer variants
- Gap: Missing discussion of models with ≤10 B parameters beyond TinyLlama (e.g., Phi‑2, MobileBERT‑LLM)
- Gap: Absence of performance metrics such as latency, perplexity, or downstream task scores for the architectures considered
- Gap: Techniques like weight pruning, mixture‑of‑experts, and detailed quantization (4‑bit, binary) are not addressed
- Gap: Impact of architecture choices on memory usage and on‑device latency is not quantified

## Sources

1. [TinyLlama/TinyLlama-1.1B-Chat-v0.1 · Hugging Face](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v0.1)
2. [Large Language Models On-Device with MediaPipe and TensorFlow Lite - Google Developers Blog](https://developers.googleblog.com/large-language-models-on-device-with-mediapipe-and-tensorflow-lite)
