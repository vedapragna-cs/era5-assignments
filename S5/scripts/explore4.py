from playwright.sync_api import sync_playwright
import pathlib, re
W = pathlib.Path("widgets").resolve()
sq = lambda s: re.sub(r'\s+',' ',s).strip()

with sync_playwright() as p:
    b=p.chromium.launch()

    # ---- WIDGET 7: curriculum sweep ----
    pg=b.new_page(viewport={"width":1400,"height":1100})
    pg.goto(f"file://{W}/widget_7_curriculum_stages.html"); pg.wait_for_timeout(500)
    print("="*72); print("WIDGET 7 — CURRICULUM: sweep across the run"); print("="*72)
    mn,mx = pg.eval_on_selector("#s5cuSlider","e=>[+e.min,+e.max]")
    for frac in [0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1.0]:
        v = mn + frac*(mx-mn)
        pg.eval_on_selector("#s5cuSlider", f"e=>{{e.value={v};e.dispatchEvent(new Event('input',{{bubbles:true}}))}}")
        pg.wait_for_timeout(120)
        print(f"\n  t={frac:.3f}  {sq(pg.inner_text('#s5cuNowTxt'))}")
        print(f"    bands: {sq(pg.inner_text('#s5cuBands'))}")
    print("\n  DIFFICULTY LADDER:"); print("   ", sq(pg.inner_text("#s5cuLadder")))
    pg.close()

    # ---- WIDGET 8: instability sweep ----
    pg=b.new_page(viewport={"width":1400,"height":1100})
    pg.goto(f"file://{W}/widget_8_mixture_shift_spike.html"); pg.wait_for_timeout(500)
    print(); print("="*72); print("WIDGET 8 — MIXTURE-SHIFT SPIKE: warmup band vs frozen embeddings"); print("="*72)
    print("  controls:", pg.eval_on_selector("#s5msSh","e=>[e.min,e.max,e.step,e.value]"),
          pg.eval_on_selector("#s5msBw","e=>[e.min,e.max,e.step,e.value]"))
    def setv(sel,v): pg.eval_on_selector(sel, f"e=>{{e.value={v};e.dispatchEvent(new Event('input',{{bubbles:true}}))}}")
    for frozen in [True,False]:
        st = pg.eval_on_selector("#s5msFrozen","e=>e.classList.contains('on')")
        if st != frozen: pg.click("#s5msFrozen"); pg.wait_for_timeout(120)
        print(f"\n  --- frozen embeddings: {'ON' if frozen else 'OFF'} (sharpness = 1.0, max) ---")
        setv("#s5msSh", 1.0)
        for bw in [0,1,2,4,8,16]:
            setv("#s5msBw", bw); pg.wait_for_timeout(120)
            print(f"    warmup band {bw:>2}B tokens -> spike {sq(pg.inner_text('#s5msSpikeV')):<9} | {sq(pg.inner_text('#s5msVerdict'))[:95]}")
    pg.close()

    # ---- WIDGET 6: reasoning effort ----
    pg=b.new_page(viewport={"width":1400,"height":1100})
    pg.goto(f"file://{W}/widget_6_reasoning_effort.html"); pg.wait_for_timeout(500)
    print(); print("="*72); print("WIDGET 6 — REASONING EFFORT: cost vs accuracy per level"); print("="*72)
    tabs = pg.eval_on_selector_all("#s5re-tabs button","e=>e.map(x=>x.innerText.replace(/\\n/g,'/'))")
    for i,t in enumerate(tabs):
        pg.eval_on_selector_all("#s5re-tabs button", f"e=>e[{i}].click()"); pg.wait_for_timeout(200)
        print(f"  {t:<22} tokens={sq(pg.inner_text('#s5re-tok')):<7} acc={sq(pg.inner_text('#s5re-acc')):<8} verify={sq(pg.inner_text('#s5re-ver')):<6} | {sq(pg.inner_text('#s5re-tokSub'))[:60]} | {sq(pg.inner_text('#s5re-accSub'))[:70]}")
    pg.close()

    # ---- WIDGET 9: scarcity ----
    pg=b.new_page(viewport={"width":1400,"height":1400})
    pg.goto(f"file://{W}/widget_9_dataset_inventory.html"); pg.wait_for_timeout(500)
    print(); print("="*72); print("WIDGET 9 — DATASET INVENTORY: scarcity card + per-slot totals"); print("="*72)
    print("  VIOLET SCARCITY CARD:"); print("   ", sq(pg.inner_text("#s5di-violet")))
    print("\n  PER-SLOT TOTAL BARS:"); print("   ", sq(pg.inner_text("#s5di-mini")))
    pg.close()

    # ---- WIDGET 5: trajectory counters ----
    pg=b.new_page(viewport={"width":1400,"height":1200})
    pg.goto(f"file://{W}/widget_5_agentic_trajectory.html"); pg.wait_for_timeout(500)
    print(); print("="*72); print("WIDGET 5 — AGENTIC TRAJECTORY: supervised vs context accounting"); print("="*72)
    for i in range(12):
        try: pg.click("#s5agNext", timeout=1200)
        except Exception: break
        pg.wait_for_timeout(90)
    print("  after full reveal:", sq(pg.inner_text("#s5agMeterRead")))
    print("  trained:", sq(pg.inner_text("#s5agTrained")), "| seen:", sq(pg.inner_text("#s5agSeen")))
    print("  stage note:", sq(pg.inner_text("#s5agStageNote"))[:220])
    print("  tab note:", sq(pg.inner_text("#s5agTabNote"))[:220])
    pg.close()
    b.close()
