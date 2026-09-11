# -*- coding: utf-8 -*-
"""
Bộ thí nghiệm lấy dữ liệu cho 9 câu hỏi trong exercises.md.

Cách dùng:
    python experiments.py --offline     # chỉ thí nghiệm cục bộ, MIỄN PHÍ
    python experiments.py --online      # chỉ thí nghiệm gọi API thật (~$0.10)
    python experiments.py               # chạy tất cả
    python experiments.py 1.1 2.1       # chạy riêng vài câu

Thí nghiệm nào gọi API thật đều được đánh dấu [API] và dùng key trong .env.
"""
import random
import statistics
import sys
import time

from template import (
    OPENAI_MODEL,
    PRICING_PER_1K_TOKENS,
    call_openai,
    chat_with_system_prompt,
    count_tokens,
    estimate_cost,
    retry_with_backoff,
    run_assistant,
)

# Windows console mặc định cp1252, không in nổi tiếng Việt
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def head(title, api=False):
    tag = " [API - tốn tiền]" if api else " [cục bộ - miễn phí]"
    print("\n" + "=" * 72)
    print(title + tag)
    print("=" * 72)


# ===========================================================================
# CÂU 1.1 — Độ nhạy của temperature                                    [API]
# ===========================================================================
def exp_1_1():
    head("CÂU 1.1 — Độ nhạy của temperature", api=True)
    prompt = "Hãy kể cho tôi một sự thật thú vị về Việt Nam."
    print(f"Prompt: {prompt}")
    print("Mỗi mức chạy 2 lần để thấy tính tái lập.\n")

    for temp in (0.0, 0.5, 1.0, 1.5):
        runs = [call_openai(prompt, temperature=temp)[0] for _ in range(2)]
        print(f"--- temperature = {temp} ---")
        for i, text in enumerate(runs, 1):
            print(f"  [lần {i}] {text}")
        print(f"  -> hai lần giống hệt nhau? {runs[0] == runs[1]}")
        print(f"  -> độ dài: {count_tokens(runs[0])} token\n")

    print("Gợi ý quan sát: tính tái lập, độ đa dạng chủ đề, và ở 1.5 thì để ý")
    print("câu có bắt đầu lủng củng hay lạc đề không.")


# ===========================================================================
# CÂU 1.2 — Temperature cho chatbot hỗ trợ khách hàng                  [API]
# ===========================================================================
def exp_1_2():
    head("CÂU 1.2 — Temperature nào cho chatbot hỗ trợ khách hàng", api=True)
    system = ("Bạn là nhân viên hỗ trợ của cửa hàng ABC. "
              "Chính sách đổi trả: trong vòng 7 ngày, còn nguyên tem mác.")
    question = "Tôi mua hàng 10 ngày trước, giờ đổi được không?"
    print(f"System: {system}")
    print(f"Hỏi:   {question}")
    print("Cùng câu hỏi, hỏi 3 lần ở mỗi mức temperature.\n")

    for temp in (0.0, 1.2):
        answers = [
            chat_with_system_prompt(system, question, temperature=temp)[0]
            for _ in range(3)
        ]
        uniq = len(set(answers))
        print(f"--- temperature = {temp} ---")
        for i, a in enumerate(answers, 1):
            print(f"  [{i}] {a}")
        print(f"  -> {uniq}/3 câu trả lời khác nhau\n")

    print("Gợi ý quan sát: ở mức cao, ba khách hỏi cùng câu có nhận cùng một")
    print("câu trả lời về chính sách không? Đó là rủi ro pháp lý, không phải")
    print("chuyện thẩm mỹ.")


# ===========================================================================
# CÂU 1.3 — Đánh đổi chi phí                                        [cục bộ]
# ===========================================================================
def exp_1_3():
    head("CÂU 1.3 — Chi phí workload 10.000 user/ngày")
    users, calls_each, out_tokens = 10_000, 3, 350
    calls = users * calls_each
    total_out = calls * out_tokens

    print(f"{users:,} user × {calls_each} call = {calls:,} call/ngày")
    print(f"{calls:,} × {out_tokens} token = {total_out:,} token output/ngày\n")

    print(f"{'Model':14s} {'$/1K out':>9s} {'$/ngày':>10s} {'$/tháng':>11s} "
          f"{'$/năm':>11s}")
    print("-" * 60)
    costs = {}
    for m in ("gpt-4o", "gpt-4o-mini"):
        price = PRICING_PER_1K_TOKENS[m]["output"]
        daily = total_out / 1000 * price
        costs[m] = daily
        print(f"{m:14s} {price:9.5f} {daily:10.2f} {daily*30:11.2f} "
              f"{daily*365:11.2f}")

    ratio_out = (PRICING_PER_1K_TOKENS["gpt-4o"]["output"]
                 / PRICING_PER_1K_TOKENS["gpt-4o-mini"]["output"])
    ratio_in = (PRICING_PER_1K_TOKENS["gpt-4o"]["input"]
                / PRICING_PER_1K_TOKENS["gpt-4o-mini"]["input"])
    diff = (costs["gpt-4o"] - costs["gpt-4o-mini"]) * 30
    print(f"\nGPT-4o đắt hơn mini: {ratio_out:.2f}x (output), "
          f"{ratio_in:.2f}x (input)")
    print(f"Chênh lệch: ${diff:,.2f}/tháng")

    # Phương án phân tầng: mini xử lý mặc định, chỉ escalate phần khó lên 4o
    print("\nPhương án phân tầng (mini lo mặc định, escalate phần khó lên 4o):")
    print(f"{'% escalate':>11s} {'$/tháng':>11s} {'tiết kiệm vs 4o':>17s}")
    print("-" * 42)
    full_4o = costs["gpt-4o"] * 30
    for pct in (0, 5, 10, 20, 50, 100):
        f = pct / 100
        c = (costs["gpt-4o"] * f + costs["gpt-4o-mini"] * (1 - f)) * 30
        print(f"{pct:10d}% {c:11.2f} {(1-c/full_4o)*100:16.1f}%")


# ===========================================================================
# CÂU 2.1 — Sức mạnh của persona                                       [API]
# ===========================================================================
def exp_2_1():
    head("CÂU 2.1 — System prompt định hình phản hồi thế nào", api=True)
    question = "Giải thích blockchain là gì?"
    personas = {
        "Giáo viên tiểu học":
            "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi.",
        "Chuyên gia tài chính":
            "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ "
            "kỹ thuật.",
    }
    print(f"Câu hỏi (giống hệt nhau): {question}\n")

    results = {}
    for label, system_prompt in personas.items():
        text, latency = chat_with_system_prompt(system_prompt, question)
        results[label] = text
        print(f"--- {label} ---")
        print(f"System: {system_prompt}\n")
        print(text)
        print(f"\n  -> {len(text.split())} từ / {count_tokens(text)} token "
              f"/ {latency:.2f}s\n")

    # Đo độ "nặng" từ vựng: tỉ lệ từ dài, và các thuật ngữ chỉ xuất hiện ở 1 bên
    print("So sánh từ vựng:")
    sets = {}
    for label, text in results.items():
        words = [w.strip(".,:;()\"'").lower() for w in text.split()]
        sets[label] = set(words)
        longish = [w for w in words if len(w) >= 8]
        print(f"  {label:22s} từ dài (>=8 ký tự): {len(longish):3d} "
              f"({len(longish)/max(1,len(words))*100:.1f}%)")
    labels = list(sets)
    only_b = sorted(sets[labels[1]] - sets[labels[0]])
    print(f"\n  Từ CHỈ xuất hiện ở '{labels[1]}' (20 từ đầu):")
    print("   ", ", ".join(only_b[:20]))


# ===========================================================================
# CÂU 2.2 — tiktoken vs đếm từ                                      [cục bộ]
# ===========================================================================
VI = ("Việt Nam là một quốc gia nằm ở khu vực Đông Nam Á, có đường bờ biển "
      "dài hơn ba nghìn cây số trải dọc theo hình chữ S quen thuộc. Nền kinh "
      "tế của đất nước đã tăng trưởng nhanh chóng trong ba thập kỷ vừa qua, "
      "chuyển dịch dần từ nông nghiệp sang công nghiệp chế biến và dịch vụ. "
      "Hà Nội là thủ đô với bề dày lịch sử hơn một nghìn năm, trong khi Thành "
      "phố Hồ Chí Minh giữ vai trò đầu tàu kinh tế phía nam. Văn hóa Việt Nam "
      "chịu ảnh hưởng sâu sắc của Nho giáo và Phật giáo, thể hiện rõ trong "
      "kiến trúc đình chùa, phong tục thờ cúng tổ tiên và các lễ hội truyền "
      "thống được tổ chức quanh năm ở khắp mọi vùng miền của cả nước.")

EN = ("Vietnam is a country located in Southeast Asia, with a coastline "
      "stretching more than three thousand kilometres along its familiar "
      "S shape. The economy has grown rapidly over the past three decades, "
      "shifting gradually from agriculture towards manufacturing and services. "
      "Hanoi is the capital, carrying more than a thousand years of history, "
      "while Ho Chi Minh City serves as the economic engine of the south. "
      "Vietnamese culture is deeply influenced by Confucianism and Buddhism, "
      "clearly visible in temple architecture, ancestor worship customs and "
      "the traditional festivals held throughout the year in every region.")


def exp_2_2():
    head("CÂU 2.2 — tiktoken vs ước lượng 'số từ / 0.75'")
    import tiktoken

    print("Phần A — đoạn tiếng Việt ~100 từ, so hai cách đếm:\n")
    print(f"{'':12s} {'số từ':>7s} {'tiktoken':>9s} {'từ/0.75':>9s} "
          f"{'lệch':>8s} {'tok/từ':>8s}")
    print("-" * 58)
    for label, txt in (("Tiếng Việt", VI), ("Tiếng Anh", EN)):
        w = len(txt.split())
        t = count_tokens(txt)
        est = w / 0.75
        print(f"{label:12s} {w:7d} {t:9d} {est:9.1f} "
              f"{(t-est)/est*100:+7.1f}% {t/w:8.2f}")

    print("\nPhần B — hai đoạn trên là bản dịch của nhau, nên so được")
    print("'cùng nội dung' thay vì 'cùng số từ':\n")
    print(f"{'encoding':14s} {'model':16s} {'VI':>6s} {'EN':>6s} {'VI/EN':>7s}")
    print("-" * 54)
    for enc_name, model in (("cl100k_base", "GPT-4 / 3.5"),
                            ("o200k_base", "GPT-4o")):
        e = tiktoken.get_encoding(enc_name)
        v, a = len(e.encode(VI)), len(e.encode(EN))
        print(f"{enc_name:14s} {model:16s} {v:6d} {a:6d} {v/a:6.2f}x")

    print("\nPhần C — cùng một chữ bị cắt khác nhau giữa hai encoding:\n")
    words = ["nghìn", "trưởng", "người", "Việt", "coastline", "Confucianism"]
    print(f"{'chữ':14s} {'cl100k_base':>12s}  {'o200k_base':>11s}")
    print("-" * 42)
    e1 = tiktoken.get_encoding("cl100k_base")
    e2 = tiktoken.get_encoding("o200k_base")
    for w in words:
        print(f"{w:14s} {len(e1.encode(w)):12d}  {len(e2.encode(w)):11d}")

    print("\nCách cắt cụ thể của chữ 'nghìn':")
    for name, e in (("cl100k_base", e1), ("o200k_base", e2)):
        ids = e.encode("nghìn")
        print(f"  {name:12s} -> {[e.decode([i]) for i in ids]}")


# ===========================================================================
# CÂU 3.1 — Streaming và time-to-first-token                           [API]
# ===========================================================================
def exp_3_1():
    head("CÂU 3.1 — Streaming cải thiện cái gì", api=True)
    from openai import OpenAI

    import os
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = ("Giải thích chi tiết cách hoạt động của thuật toán "
              "quicksort, kèm ví dụ từng bước.")
    print(f"Prompt (câu trả lời dài): {prompt}\n")

    # --- không streaming: người dùng chờ trắng màn hình đến tận lúc xong ---
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
    )
    non_stream_total = time.perf_counter() - t0
    text_ns = resp.choices[0].message.content

    # --- streaming: đo riêng lúc chữ ĐẦU TIÊN xuất hiện ---
    t0 = time.perf_counter()
    stream = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
        stream=True,
    )
    ttft, text_s, n_chunks = None, "", 0
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta and ttft is None:
            ttft = time.perf_counter() - t0
        text_s += delta
        n_chunks += 1
    stream_total = time.perf_counter() - t0

    print(f"{'':22s} {'chữ đầu tiên':>14s} {'hoàn tất':>10s}")
    print("-" * 50)
    print(f"{'Không streaming':22s} {non_stream_total:13.2f}s "
          f"{non_stream_total:9.2f}s")
    print(f"{'Streaming':22s} {ttft:13.2f}s {stream_total:9.2f}s")
    print(f"\nNgười dùng thấy chữ đầu sớm hơn "
          f"{non_stream_total - ttft:.2f}s "
          f"({non_stream_total / max(ttft, 1e-9):.1f}x nhanh hơn về cảm giác)")
    print(f"Tổng thời gian gần như không đổi: "
          f"{non_stream_total:.2f}s vs {stream_total:.2f}s")
    print(f"Stream về trong {n_chunks} chunk, "
          f"{count_tokens(text_s)} token")
    print("\nGợi ý quan sát: streaming KHÔNG làm API nhanh hơn — nó chỉ đổi")
    print("thời điểm người dùng bắt đầu đọc được. Đó là lý do nó vô nghĩa khi")
    print("output dùng cho máy (parse JSON) chứ không cho người.")


# ===========================================================================
# CÂU 3.2 — Backoff, thundering herd, jitter                        [cục bộ]
# ===========================================================================
def exp_3_2():
    head("CÂU 3.2 — Vì sao exponential backoff, và vì sao cần jitter")

    # --- A. retry_with_backoff thật sự chờ bao lâu ---
    print("Phần A — đo retry_with_backoff của bài (base_delay=0.1):\n")
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 4:
            raise ConnectionError(f"lỗi giả lần {calls['n']}")
        return "thành công"

    t0 = time.perf_counter()
    result = retry_with_backoff(flaky, max_retries=3, base_delay=0.1)
    elapsed = time.perf_counter() - t0
    print(f"  Kết quả: {result} sau {calls['n']} lần gọi, {elapsed:.2f}s")
    print(f"  Delay cộng dồn kỳ vọng: 0.1 + 0.2 + 0.4 = 0.70s\n")

    # --- B. tổng thời gian chờ: cố định vs cấp số nhân ---
    print("Phần B — chờ bao lâu sau N lần thử:\n")
    print(f"{'lần thử':>8s} {'delay cố định 1s':>18s} {'exponential':>13s}")
    print("-" * 42)
    for n in range(1, 7):
        fixed = n * 1.0
        expo = sum(0.1 * 2 ** i for i in range(n))
        print(f"{n:8d} {fixed:17.1f}s {expo:12.1f}s")

    # --- C. mô phỏng thundering herd ---
    # Đo ở độ phân giải 0.25s: server quá tải quan tâm request/giây tức thời,
    # không phải tổng theo từng giây. Gộp theo giây sẽ giấu mất các spike.
    print("\nPhần C — mô phỏng 2000 client cùng gặp lỗi tại t=0.")
    print("Đo ở độ phân giải 0.25s (ô 1 giây quá thô, che mất spike).\n")
    random.seed(42)
    n_clients, horizon, res = 2000, 16.0, 0.25
    n_buckets = int(horizon / res)

    def simulate(strategy):
        buckets = [0] * n_buckets
        for _ in range(n_clients):
            t = 0.0
            for attempt in range(5):
                step = 1.0 * 2 ** attempt
                if strategy == "fixed":
                    t += 1.0
                elif strategy == "expo":
                    t += step
                else:                       # full jitter
                    t += random.uniform(0, step)
                if t < horizon:
                    buckets[int(t / res)] += 1
        return buckets

    names = {"fixed": "Delay cố định 1s", "expo": "Exponential",
             "jitter": "Exponential + jitter"}
    order = ("fixed", "expo", "jitter")
    sims = {s: simulate(s) for s in order}
    peak = max(max(b) for b in sims.values())

    for s in order:
        print(f"  {names[s]} — dòng thời gian 0→{horizon:.0f}s, "
              f"mỗi ký tự = {res}s:")
        # mỗi bucket thành 1 ký tự, độ đậm tỉ lệ với tải
        marks = " .:-=+*#@"
        bar = "".join(marks[min(len(marks) - 1, int(v / peak * (len(marks) - 1)))]
                      if v else " " for v in sims[s])
        print(f"    |{bar}|")
        print(f"    đỉnh {max(sims[s]):4d} req/{res}s, "
              f"tổng {sum(sims[s]):5d} req\n")

    print(f"  {'Chiến lược':24s} {'đỉnh req/0.25s':>15s} {'so với jitter':>15s}")
    print("  " + "-" * 56)
    jitter_peak = max(sims["jitter"])
    for s in order:
        p = max(sims[s])
        print(f"  {names[s]:24s} {p:15d} {p / jitter_peak:14.1f}x")

    print("\nGợi ý quan sát: delay cố định tạo sóng ĐỀU ĐẶN không hề giảm —")
    print("server quá tải không bao giờ được nghỉ. Exponential giãn sóng thưa")
    print("dần theo thời gian, nhưng vì mọi client dùng chung một lịch nên mỗi")
    print("sóng vẫn dồn cục đúng một thời điểm, đỉnh vẫn cao y hệt. Chỉ jitter")
    print("mới hạ được ĐỈNH tải — mà đỉnh mới là thứ làm sập server.")
    print("retry_with_backoff trong bài CHƯA có jitter.")


# ===========================================================================
# CÂU 4.1 + 4.2 — Trợ lý CLI: persona và giới hạn bộ nhớ               [API]
# ===========================================================================
def exp_4():
    head("CÂU 4.1 & 4.2 — Persona và giới hạn nhớ 3 lượt", api=True)

    persona = ("Bạn là trợ giảng thân thiện của khóa AI, "
               "trả lời ngắn gọn bằng tiếng Việt.")
    print(f"Persona: {persona}\n")

    # Kịch bản cố ý: khai tên ở lượt 1, hỏi lại ở lượt 6 (đã bị cắt khỏi history)
    script = [
        "Mình tên Giáp, đang học Python. Nhớ giúp mình nhé.",
        "Temperature trong LLM là gì?",
        "Còn top_p thì sao?",
        "Streaming có lợi gì?",
        "Token là gì?",
        "Mình tên gì và đang học ngôn ngữ nào?",   # <- phép thử trí nhớ
    ]
    it = iter(script)

    def scripted_input():
        msg = next(it)
        print(f"\n\033[1mBạn:\033[0m {msg}\n\033[1mTrợ lý:\033[0m ", end="")
        return msg

    stats = run_assistant(persona, get_input=scripted_input,
                          max_turns=len(script))

    print("\n\n--- Thống kê phiên ---")
    print(f"  số lượt      : {stats['num_turns']}")
    print(f"  tổng token   : {stats['total_tokens']:,}")
    print(f"  tổng chi phí : ${stats['total_cost']:.6f}")
    print(f"  history còn  : {len(stats['history'])} message "
          f"(tối đa 6 = 3 lượt)")

    print("\n--- History còn lại sau khi cắt ---")
    for m in stats["history"]:
        print(f"  [{m['role']:9s}] {m['content'][:60]}...")

    print("\nGợi ý quan sát cho Câu 4.2: lượt cuối hỏi lại tên. Tin nhắn khai")
    print("tên ở lượt 1 đã bị history[-6:] cắt mất, nên trợ lý phải đoán hoặc")
    print("nhận là không biết. Đó chính là hạn chế cần đề xuất cải thiện.")

    # Chi phí thật của phiên so với nếu gửi TOÀN BỘ history mỗi lượt
    print("\n--- Vì sao phải cắt history (chi phí) ---")
    price = PRICING_PER_1K_TOKENS.get(OPENAI_MODEL,
                                      PRICING_PER_1K_TOKENS["gpt-4o"])
    for n_turns in (3, 10, 30, 100):
        avg_msg = 60  # token trung bình mỗi message
        capped = n_turns * (6 * avg_msg) / 1000 * price["input"]
        full = sum(2 * i * avg_msg for i in range(n_turns)) / 1000 * price["input"]
        print(f"  {n_turns:3d} lượt: cắt 3 lượt ${capped:.4f}  |  "
              f"gửi hết ${full:.4f}  ({full/max(capped,1e-9):.1f}x)")


# ===========================================================================
EXPERIMENTS = {
    "1.1": (exp_1_1, True),
    "1.2": (exp_1_2, True),
    "1.3": (exp_1_3, False),
    "2.1": (exp_2_1, True),
    "2.2": (exp_2_2, False),
    "3.1": (exp_3_1, True),
    "3.2": (exp_3_2, False),
    "4": (exp_4, True),
}


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    flags = {a for a in argv[1:] if a.startswith("--")}

    if args:
        keys = [a for a in args if a in EXPERIMENTS]
        unknown = [a for a in args if a not in EXPERIMENTS]
        if unknown:
            print(f"Không nhận ra: {', '.join(unknown)}")
            print(f"Chọn trong: {', '.join(EXPERIMENTS)}")
            return 1
    elif "--offline" in flags:
        keys = [k for k, (_, api) in EXPERIMENTS.items() if not api]
    elif "--online" in flags:
        keys = [k for k, (_, api) in EXPERIMENTS.items() if api]
    else:
        keys = list(EXPERIMENTS)

    uses_api = [k for k in keys if EXPERIMENTS[k][1]]
    if uses_api:
        print(f"Sắp chạy {len(uses_api)} thí nghiệm GỌI API THẬT "
              f"({', '.join(uses_api)}) bằng key trong .env.")
        print("Ước tính dưới $0.15 với gpt-4o. Ctrl+C để huỷ.")
        print("Chạy `python experiments.py --offline` nếu chỉ muốn phần miễn phí.")
        try:
            time.sleep(3)
        except KeyboardInterrupt:
            print("\nĐã huỷ.")
            return 130

    for k in keys:
        fn, _ = EXPERIMENTS[k]
        try:
            fn()
        except Exception as exc:            # một thí nghiệm hỏng không chặn phần còn lại
            print(f"\n!! Thí nghiệm {k} lỗi: {type(exc).__name__}: {exc}")

    print("\n" + "=" * 72)
    print("Xong. Đọc output rồi tự viết nhận xét vào exercises.md:")
    print("  Câu 1.1 -> dòng 18      Câu 2.1 -> dòng 69")
    print("Bảy câu còn lại đã có sẵn câu trả lời, đọc lại và sửa theo ý bạn.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
