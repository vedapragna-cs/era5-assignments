from playwright.sync_api import sync_playwright
import pathlib, statistics as st
W = pathlib.Path("widgets").resolve()

def pct(s): return float(s.replace("%","").strip())

with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={"width":1400,"height":1100})
    pg.goto(f"file://{W}/widget_3_opus_selection.html"); pg.wait_for_timeout(500)

    print("="*72); print("WIDGET 3 — OPUS: keep-fraction economics"); print("="*72)
    for k in [10,20,30,40,50,60,70,80,90]:
        pg.eval_on_selector("#s5opKeep", f"e=>{{e.value={k};e.dispatchEvent(new Event('input',{{bubbles:true}}))}}")
        pg.wait_for_timeout(120)
        print(f"  keep {k:>2}% -> mult {pg.inner_text('#s5opMult'):<7} | {pg.inner_text('#s5opTokLine')}")

    # reset keep to V4 production value
    pg.eval_on_selector("#s5opKeep", "e=>{e.value=40;e.dispatchEvent(new Event('input',{bubbles:true}))}")

    print()
    print("="*72); print("MEASURED: share of TRAINED tokens per iteration, N=60 iterations each"); print("="*72)
    for proxy in ["s5opProxEng","s5opProxBal"]:
        for lane_on in [False, True]:
            pg.click("#"+proxy); pg.wait_for_timeout(120)
            # set lane switch to desired state
            state = pg.eval_on_selector("#s5opLane","e=>e.classList.contains('on')")
            if state != lane_on:
                pg.click("#s5opLane"); pg.wait_for_timeout(120)
            ind, ag = [], []
            for _ in range(60):
                pg.click("#s5opNext"); pg.wait_for_timeout(35)
                ind.append(pct(pg.inner_text("#s5opIndicVal")))
                ag.append(pct(pg.inner_text("#s5opAgVal")))
            name = "English-heavy(V4)" if proxy=="s5opProxEng" else "Balanced"
            print(f"\n  proxy={name:<18} always-on={'ON ' if lane_on else 'OFF'}")
            print(f"    Indic   share of trained tokens: mean {st.mean(ind):5.2f}%  median {st.median(ind):5.2f}%  min {min(ind):5.2f}%  max {max(ind):5.2f}%  zero-iters {sum(1 for x in ind if x==0)}/60")
            print(f"    Agentic share of trained tokens: mean {st.mean(ag):5.2f}%  median {st.median(ag):5.2f}%  min {min(ag):5.2f}%  max {max(ag):5.2f}%  zero-iters {sum(1 for x in ag if x==0)}/60")
            print(f"    lane note: {pg.inner_text('#s5opLaneNote')[:160]}")
    b.close()
