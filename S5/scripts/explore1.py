from playwright.sync_api import sync_playwright
import pathlib, re
W = pathlib.Path("widgets").resolve()
def clean(s): return re.sub(r'\n{2,}','\n', s).strip()

with sync_playwright() as p:
    b = p.chromium.launch()

    # ---------- WIDGET 1: preset x run size ----------
    pg = b.new_page(viewport={"width":1400,"height":1100})
    pg.goto(f"file://{W}/widget_1_mixture_composer.html"); pg.wait_for_timeout(600)
    print("="*70); print("WIDGET 1 — MIXTURE COMPOSER: preset x run-size supply matrix"); print("="*70)
    for preset in ["pretrain","anneal","naive"]:
        pg.click(f"#s5mx-preset-{preset}"); pg.wait_for_timeout(250)
        bar = pg.inner_text("#s5mx-legend").replace("\n"," ")
        print(f"\n--- PRESET: {preset} ---")
        print("  mix:", re.sub(r'\s+',' ',bar))
        print("  readout:", re.sub(r'\s+',' ',pg.inner_text("#s5mx-readout")))
        print("  floornote:", re.sub(r'\s+',' ',pg.inner_text("#s5mx-floornote"))[:300])
        for run in [1,2,5,10]:
            pg.click(f'#s5mx-runsel button[data-run="{run}"]'); pg.wait_for_timeout(200)
            rows = pg.eval_on_selector_all("#s5mx-supplybody .s5mx-suprow",
                "els=>els.map(e=>e.innerText.replace(/\\n/g,' | '))")
            print(f"  [run {run}T]")
            for r in rows: print("     ", re.sub(r'\s+',' ',r))
    pg.close()
    b.close()
