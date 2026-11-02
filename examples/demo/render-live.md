# State of Small Language Models for On‑Device Use (2026)

**Summary**  
Small language models (SLMs)—defined as models under 2 billion parameters—have matured to run on a wide range of low‑power hardware, from smartphones to edge devices such as Raspberry Pi 5 and smart‑watch class chips. By 2025 the practical on‑device limit expanded to ~14 billion parameters when 4‑bit quantization is applied, while sub‑7 billion models comfortably operate on modern phones and sub‑3 billion models on more constrained edge platforms [3]. In 2026 SLMs now support reasoning, tool‑calling, multimodal inputs, long‑document processing, and partial agentic workflows, narrowing the gap with larger LLMs [2]. Industry adoption is accelerating, driven by privacy, latency, and cost incentives across health, finance, and manufacturing sectors [1][4].

## Background and Current State  

Small language models (SLMs) are AI models with fewer than 2 billion parameters that can run efficiently on low‑power devices, offering reduced memory use, energy consumption, and deployment costs [1]. The practical on‑device threshold has risen: with 4‑bit quantization, models up to roughly 14 billion parameters can run on a machine equipped with 16 GB RAM, while models under 7 billion parameters run on contemporary smartphones and those under 3 billion run on edge hardware such as Raspberry Pi 5 or smart‑watch class devices [3].  

In 2026 SLMs have progressed to support advanced capabilities—including reasoning, tool‑calling, multimodal (image and audio) inputs, long‑document processing, and components of agentic workflows—thereby narrowing the performance gap with larger models [2]. Prominent releases in 2024‑2025, such as Phi‑3‑mini (3.8 B), Gemma 2 (2 B/9 B) and Llama 3.2 (1 B/3 B), demonstrate that models in the 3–7 billion‑parameter range can match or exceed much larger models on focused tasks while running locally on laptops, phones, or edge devices at zero API cost [3].  

Industry reports from 2025 highlight SLMs as a key enabler for real‑time, resource‑limited applications across health, finance, and manufacturing, with over 100 companies evaluated and 26 recognized as market leaders [1]. The broader edge computing market is projected to reach $378 billion by 2028, and 75 % of enterprise data is expected to be processed at the edge by 2025, creating strong demand for privacy‑preserving AI that operates on modest hardware [4].

## Limits, Risks and Outlook  

Memory remains the primary bottleneck for on‑device SLMs, limiting both model size and prompting length despite advances in mobile NPUs [9]. Current NPUs—such as Apple A19 Pro (~35 TOPS) and Qualcomm Snapdragon 8 Elite Gen 5 (~60 TOPS)—often lack support for essential operations like attention and dynamic shapes, complicating SLM deployment [9].  

On‑device inference can achieve sub‑20 ms token latency for short contexts, but longer contexts or frontier reasoning tasks still rely on cloud resources [9]. Privacy guarantees can be weakened by insecure toolchains, side‑channel attacks, or inadvertent memorization of sensitive data within the models [9]. Power and thermal budgets of edge devices also restrict continuous high‑throughput SLM use, making always‑on agents impractical without aggressive scaling or duty‑cycling strategies [9].  

The future outlook anticipates tighter model compression (quantization, pruning) combined with next‑generation NPUs, which together may close much of the capability gap between small and large language models for many agentic tasks [8][9]. Heterogeneous agentic systems that dynamically switch between on‑device SLMs and larger cloud models are projected to become the dominant architecture for balancing cost, latency, and capability [8].

## Missing Sections  

The “Numbers and evidence” section could not be populated because the source material provided no usable findings (pages were fetched but no findings could be extracted). No content is presented for this section.

## Coverage notes

- Missing: "Numbers and evidence" has no usable findings (pages were fetched but no findings could be extracted (model unavailable or unusable replies)).

**Claim check:** 13 claims cited, 0 claims left out or marked unverified, 1 sections missing or weak.

## Sources

1. [Small Language Models (SMLs) Company Evaluation Report 2025 | Azure AI Integration, Watson Platform Advancements, and On-Device Processing Strengthen Leadership in Health, Finance, Manufacturing](https://finance.yahoo.com/news/small-language-models-smls-company-090100930.html)
2. [Small Language Models in 2026: 10 to Know, Comparedcaret-right](https://www.turingpost.com/p/slmslist)
3. [Small Language Models — Phi, Gemma, On-Device LLM Patterns — ResumeLens AI Blog](https://www.resumelens.org/blog/ai/small-language-models-edge)
4. [Small Language Models: Deploying Efficient AI at the Edge](https://www.runpod.io/articles/guides/small-language-models-revolution-deploying-efficient-ai-at-the-edge)
5. [Best Small Language Models (2026): SLMs Compared by Task](https://vdf.ai/blog/best-small-language-models)
6. [On-Device SLMs: Why Benchmarks Miss Real Phone Performance · AlterSquare](https://altersquare.io/blog/on-device-ai-slms-latency-memory-tradeoffs)
7. [Best Small Language Models 2026: Top SLMs Ranked (1B-14B)](https://localaimaster.com/blog/small-language-models-guide-2026)
8. [Small Language Models are the Future of Agentic AI](https://arxiv.org/html/2506.02153v3)
9. [On-Device LLMs: State of the Union, 2026 – Vikas Chandra – VP, AI @ Atoms](https://v-chandra.github.io/on-device-llms)
10. [On-Device LLMs: Complete Guide to Local AI (2026)](https://picovoice.ai/blog/on-device-llm)
