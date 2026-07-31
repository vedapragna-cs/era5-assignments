from playwright.sync_api import sync_playwright
import pathlib, re
W = pathlib.Path("widgets").resolve()
sq = lambda s: re.sub(r'\s+',' ',s).strip()
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={"width":1400,"height":1100})
    pg.goto(f"file://{W}/widget_8_mixture_shift_spike.html"); pg.wait_for_timeout(500)
    def setv(sel,v): pg.eval_on_selector(sel, f"e=>{{e.value={v};e.dispatchEvent(new Event('input',{{bubbles:true}}))}}")
    print("="*72); print("WIDGET 8 — spike multiplier vs warmup band and frozen embeddings"); print("="*72)
    for frozen in [True, False]:
        for sh in [1.0, 0.7]:
            pg.click("#s5msReset"); pg.wait_for_timeout(150)
            st = pg.eval_on_selector("#s5msFrozen","e=>e.classList.contains('on')")
            if st != frozen: pg.click("#s5msFrozen"); pg.wait_for_timeout(100)
            setv("#s5msSh", sh)
            print(f"\n  --- frozen={'ON ' if frozen else 'OFF'}  sharpness={sh} ---")
            for bw in [0,0.5,1,2,3,4,5]:
                setv("#s5msBw", bw); pg.wait_for_timeout(80)
                pg.click("#s5msApply"); pg.wait_for_timeout(150)
                print(f"    band {bw:>4}B -> spike {sq(pg.inner_text('#s5msSpikeV')):<8} "
                      f"stat={sq(pg.inner_text('#s5msSpikeS')):<12} | {sq(pg.inner_text('#s5msVerdict'))[:110]}")
    print("\n  READBOX:", sq(pg.inner_text("#s5msReadBox"))[:400])
    b.close()
