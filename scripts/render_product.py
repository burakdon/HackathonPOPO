"""Rebuild product/index.html and product/page.html from product/template.html and the existing mock data.

Use this after editing the template. It does not rerun the simulator: scripts/build_product_mock.py
regenerates product/mock/data.json too, which is slow and changes the data.

Usage:  uv run python -m scripts.render_product
"""
from __future__ import annotations

import base64
import json

from core.config import resolve

WRAPPER = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
           '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
           '<style>[hidden]{display:none!important}img{max-width:100%}</style>\n</head>\n<body>\n')


def main() -> None:
    data = json.loads(resolve("product/mock/data.json").read_text())
    items = json.loads(resolve("data/items.json").read_text())["items"]
    pics = {it["exercise_id"]: "data:image/svg+xml;base64,"
            + base64.b64encode((resolve("frontend") / it["image_path"]).read_bytes()).decode() for it in items}
    tpl = resolve("product/template.html").read_text()
    html = (tpl.replace("/*__MOCK_DATA__*/null", json.dumps(data, separators=(",", ":")))
               .replace("/*__ITEMS__*/null", json.dumps(items, separators=(",", ":")))
               .replace("/*__PICTURES__*/null", json.dumps(pics, separators=(",", ":"))))
    resolve("product/page.html").write_text(html)
    resolve("product/index.html").write_text(WRAPPER + html + "\n</body>\n</html>\n")
    print(f"wrote product/index.html and product/page.html ({len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
