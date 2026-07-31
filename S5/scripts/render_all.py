from playwright.sync_api import sync_playwright
import pathlib, os
d = pathlib.Path("widgets")
out = pathlib.Path("rendered"); out.mkdir(exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch()
    for f in sorted(d.glob("*.html")):
        pg = b.new_page(viewport={"width":1280,"height":1000})
        errs=[]
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto("file://"+str(f.resolve()))
        pg.wait_for_timeout(1200)
        txt = pg.inner_text("body")
        (out/(f.stem+".txt")).write_text(txt)
        pg.screenshot(path=f"shots/{f.stem}.png", full_page=True)
        print(f"{f.stem}: {len(txt)} chars, errors={errs[:2]}")
        pg.close()
    b.close()
