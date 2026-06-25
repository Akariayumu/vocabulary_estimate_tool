"""将本估算器与 testyourvocab.com 的浏览器运行结果对比。

外部网站会随时间变化，也可能重定向到 Preply。因此本模块将浏览器自动化视为
best-effort：记录最终 URL，处理重定向，并仍然计算本地算法结果用于对比。
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from vocab_estimator.config import DEFAULT_CONFIG
from vocab_estimator.vocab_bank import VocabBank
from vocab_estimator.vocab_model import VocabEstimator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "browser_verification_report.json"


def load_known_words(value: str | None) -> set[str]:
    """从逗号分隔字符串或文本文件加载已知词。"""

    if not value:
        return {
            "the",
            "school",
            "student",
            "language",
            "computer",
            "analysis",
            "important",
            "possible",
            "develop",
            "evidence",
            "strategy",
        }
    path = Path(value)
    if path.exists():
        text = path.read_text(encoding="utf-8")
        return {line.strip().lower() for line in text.splitlines() if line.strip()}
    return {word.strip().lower() for word in value.split(",") if word.strip()}


def local_estimate(known_words: set[str]) -> dict[str, Any]:
    """用词库中可见的已知词运行本项目估算器。"""

    bank = VocabBank(DEFAULT_CONFIG)
    responses = [(item.word, item.word.lower() in known_words) for item in bank.items[:160]]
    result = VocabEstimator(bank, DEFAULT_CONFIG).estimate_single(responses)
    return {
        "result": result,
        "response_count": len(responses),
        "known_count": sum(known for _word, known in responses),
    }


def api_estimate(api_url: str, known_words: set[str]) -> dict[str, Any] | None:
    """按需调用正在运行的本地 API。"""

    bank = VocabBank(DEFAULT_CONFIG)
    responses = [
        {"word": item.word, "known": item.word.lower() in known_words}
        for item in bank.items[:160]
    ]
    request = Request(
        f"{api_url.rstrip('/')}/api/estimate",
        data=json.dumps({"responses": responses}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except URLError:
        return None


def run_browser(site_url: str, known_words: set[str], headless: bool = True) -> dict[str, Any]:
    """打开外部网站，尽可能勾选已知词，并抓取结果。"""

    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # pragma: no cover - 依赖可能未安装。
        return {
            "status": "playwright_unavailable",
            "error": str(exc),
            "site_url": site_url,
        }

    report: dict[str, Any] = {
        "status": "started",
        "site_url": site_url,
        "final_url": None,
        "checked_words": [],
        "site_result": None,
    }

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        try:
            page.goto(site_url, wait_until="domcontentloaded", timeout=45000)
            page.wait_for_timeout(1500)
            report["final_url"] = page.url

            if "preply" in page.url.lower():
                report["status"] = "redirected_to_preply"
                report["message"] = "testyourvocab.com redirected to Preply; external score unavailable."
                return report

            checked = check_known_words(page, known_words)
            report["checked_words"] = checked
            submit_visible_form(page)
            page.wait_for_timeout(2500)
            report["final_url"] = page.url
            report["site_result"] = scrape_vocab_result(page)
            report["status"] = "completed" if report["site_result"] else "completed_without_result"
            return report
        except PlaywrightTimeoutError as exc:
            report["status"] = "timeout"
            report["error"] = str(exc)
            return report
        except Exception as exc:
            report["status"] = "error"
            report["error"] = str(exc)
            return report
        finally:
            browser.close()


def check_known_words(page: Any, known_words: set[str]) -> list[str]:
    """勾选相邻标签匹配已知词的可见复选框。"""

    script = """
    (knownWords) => {
      const known = new Set(knownWords.map(w => w.toLowerCase()));
      const checked = [];
      const normalize = (s) => (s || '').trim().toLowerCase();

      for (const label of Array.from(document.querySelectorAll('label'))) {
        const text = normalize(label.innerText || label.textContent);
        if (!known.has(text)) continue;
        let input = null;
        if (label.htmlFor) input = document.getElementById(label.htmlFor);
        if (!input) input = label.querySelector('input[type="checkbox"]');
        if (!input) {
          const prev = label.previousElementSibling;
          const next = label.nextElementSibling;
          if (prev && prev.matches && prev.matches('input[type="checkbox"]')) input = prev;
          if (!input && next && next.matches && next.matches('input[type="checkbox"]')) input = next;
        }
        if (input && input.type === 'checkbox' && !input.checked) {
          input.click();
          checked.push(text);
        }
      }

      for (const input of Array.from(document.querySelectorAll('input[type="checkbox"]'))) {
        const parentText = normalize(input.parentElement && input.parentElement.innerText);
        if (known.has(parentText) && !input.checked) {
          input.click();
          checked.push(parentText);
        }
      }

      return Array.from(new Set(checked));
    }
    """
    return page.evaluate(script, sorted(known_words))


def submit_visible_form(page: Any) -> None:
    """点击最可能的提交/继续按钮。"""

    candidates = [
        "button[type=submit]",
        "input[type=submit]",
        "button:has-text('Submit')",
        "button:has-text('Continue')",
        "button:has-text('Next')",
        "a:has-text('Submit')",
        "a:has-text('Continue')",
    ]
    for selector in candidates:
        locator = page.locator(selector).first
        try:
            if locator.count() and locator.is_visible():
                locator.click(timeout=3000)
                return
        except Exception:
            continue


def scrape_vocab_result(page: Any) -> int | None:
    """在页面文本中查找词汇量数字。"""

    text = page.locator("body").inner_text(timeout=5000)
    patterns = [
        r"vocabulary\s+size\s+(?:is|of)?\s*([0-9][0-9,]{3,})",
        r"you\s+know\s+(?:about\s+)?([0-9][0-9,]{3,})",
        r"([0-9][0-9,]{3,})\s+words",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1).replace(",", ""))
    return None


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Browser-compare estimator with testyourvocab.com.")
    parser.add_argument("--site-url", default="https://testyourvocab.com/", help="External test site URL.")
    parser.add_argument("--known-words", help="Comma-separated words or a text file path.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000", help="Optional local API URL.")
    parser.add_argument("--headed", action="store_true", help="Run browser visibly instead of headless.")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT, help="Report JSON path.")
    args = parser.parse_args()

    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    known_words = load_known_words(args.known_words)
    browser_report = run_browser(args.site_url, known_words, headless=not args.headed)
    local = local_estimate(known_words)
    api = api_estimate(args.api_url, known_words)

    site_result = browser_report.get("site_result")
    local_point = local["result"]["point_estimate"]
    report = {
        "created_at": started_at,
        "known_words": sorted(known_words),
        "browser": browser_report,
        "local_algorithm": local,
        "api_algorithm": api,
        "comparison": {
            "site_result": site_result,
            "local_point_estimate": local_point,
            "difference_site_minus_local": site_result - local_point if isinstance(site_result, int) else None,
        },
    }

    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    write_report(report, output)
    print(f"wrote: {display_path(output)}")
    print(f"browser_status: {browser_report.get('status')}")
    print(f"local_point_estimate: {local_point}")
    if site_result is not None:
        print(f"site_result: {site_result}")


if __name__ == "__main__":
    main()
