"""Capture dashboard screenshots with Playwright (requires API + Vite running)."""
import asyncio
import time
from pathlib import Path

from playwright.async_api import async_playwright

OUT = Path(__file__).resolve().parents[1] / "docs"
OUT.mkdir(exist_ok=True)


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1440, "height": 900})
        await page.goto("http://127.0.0.1:5173/", wait_until="networkidle", timeout=60000)
        # Wait for live feed / metrics to populate
        await page.wait_for_timeout(8000)
        await page.screenshot(path=str(OUT / "dashboard_main.png"), full_page=False)
        print("saved", OUT / "dashboard_main.png")

        await page.get_by_role("button", name="Threat Intelligence").click()
        await page.wait_for_timeout(4000)
        await page.screenshot(path=str(OUT / "dashboard_threat_intel.png"), full_page=False)
        print("saved", OUT / "dashboard_threat_intel.png")

        # Short live-feed GIF via frames on main dashboard
        await page.get_by_role("button", name="Dashboard").click()
        await page.wait_for_timeout(1500)
        frames_dir = OUT / "_gif_frames"
        frames_dir.mkdir(exist_ok=True)
        frame_paths = []
        for i in range(24):
            fp = frames_dir / f"frame_{i:02d}.png"
            await page.screenshot(path=str(fp), full_page=False)
            frame_paths.append(fp)
            await page.wait_for_timeout(1000)
        # Try pillow GIF
        try:
            from PIL import Image
            imgs = [Image.open(fp).convert("P") for fp in frame_paths]
            gif_path = OUT / "live_feed.gif"
            imgs[0].save(
                gif_path,
                save_all=True,
                append_images=imgs[1:],
                duration=400,
                loop=0,
                optimize=True,
            )
            print("saved", gif_path)
        except Exception as e:
            print("GIF_SKIPPED", type(e).__name__, e)

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
