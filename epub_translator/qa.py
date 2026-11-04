"""QA: 1) structure checks on the output EPUB, 2) translation checks, EPUBCheck runner, JSON/HTML report."""
from __future__ import annotations
import html
import json
import os
import posixpath
import re
import shutil
import subprocess
import zipfile
from collections import Counter, defaultdict

from lxml import etree

from .epub_model import EpubBook, NS, XHTML, parse_xml
from .textutil import is_dialogue, strip_tokens, norm, HANGUL
from .validator import check_block, find_duplicates, target_quote_count


def structure_check(path: str) -> dict:
    res = {"ok": True, "errors": [], "warnings": [], "checked": {}}

    def err(m):
        res["ok"] = False
        res["errors"].append(m)

    try:
        z = zipfile.ZipFile(path)
    except zipfile.BadZipFile as e:
        err(f"ZIP 손상: {e}")
        return res
    infos = z.infolist()
    if not infos or infos[0].filename != "mimetype":
        err("mimetype 이 ZIP 첫 항목이 아님")
    elif infos[0].compress_type != zipfile.ZIP_STORED:
        err("mimetype 이 압축되어 있음")
    elif z.read("mimetype").strip() != b"application/epub+zip":
        err("mimetype 내용 오류")
    bad = z.testzip()
    if bad:
        err(f"CRC 오류: {bad}")
    try:
        book = EpubBook(path)
    except Exception as e:
        err(f"OPF/container 파싱 실패: {e}")
        return res
    names = set(z.namelist())
    missing = [m["path"] for m in book.manifest.values() if m["path"] not in names]
    for m in missing:
        err(f"manifest 파일 누락: {m}")
    res["checked"]["manifest_items"] = len(book.manifest)
    # XHTML parse + links + images
    ids = defaultdict(set)
    docs = {}
    parse_errors = 0
    for p in book.xhtml_items():
        try:
            root = etree.fromstring(z.read(p), etree.XMLParser(recover=False, resolve_entities=False, huge_tree=True))
            docs[p] = root
            for el in root.iter():
                if isinstance(el.tag, str) and el.get("id"):
                    ids[p].add(el.get("id"))
        except etree.XMLSyntaxError as e:
            parse_errors += 1
            err(f"XHTML 파싱 오류 {p}: {e}")
    res["checked"]["xhtml_files"] = len(docs)
    res["checked"]["xhtml_parse_errors"] = parse_errors
    broken_links, imgs, missing_imgs = 0, 0, 0
    for p, root in docs.items():
        base = posixpath.dirname(p)
        for el in root.iter():
            if not isinstance(el.tag, str):
                continue
            for attr in ("src", "{http://www.w3.org/1999/xlink}href", "href"):
                v = el.get(attr)
                if not v or re.match(r"^[a-z]+:", v, re.I):
                    continue
                target = book.resolve(base, v) if not v.startswith("#") else p
                frag = v.split("#", 1)[1] if "#" in v else None
                tag = el.tag.split("}")[-1]
                if tag in ("img", "image"):
                    imgs += 1
                    if target not in names:
                        missing_imgs += 1
                        err(f"이미지 누락: {p} -> {v}")
                elif tag == "a":
                    if target not in names:
                        broken_links += 1
                        err(f"깨진 링크: {p} -> {v}")
                    elif frag and target in docs and frag not in ids[target]:
                        broken_links += 1
                        res["warnings"].append(f"앵커 없음: {p} -> {v}")
    res["checked"].update(images=imgs, missing_images=missing_imgs, broken_links=broken_links)
    # TOC links
    toc_links, toc_broken = 0, 0
    if book.ncx_path:
        ncx = parse_xml(z.read(book.ncx_path))
        b = posixpath.dirname(book.ncx_path)
        for c in ncx.iter("{%s}content" % NS["ncx"]):
            toc_links += 1
            t = book.resolve(b, c.get("src", ""))
            frag = c.get("src", "").split("#", 1)[1] if "#" in c.get("src", "") else None
            if t not in names or (frag and t in docs and frag not in ids[t]):
                toc_broken += 1
                err(f"NCX 목차 링크 오류: {c.get('src')}")
    if book.nav_path:
        nav = docs[book.nav_path] if book.nav_path in docs else parse_xml(z.read(book.nav_path))
        b = posixpath.dirname(book.nav_path)
        for a in nav.iter(XHTML + "a"):
            h = a.get("href", "")
            toc_links += 1
            t = book.resolve(b, h)
            frag = h.split("#", 1)[1] if "#" in h else None
            if t not in names or (frag and t in docs and frag not in ids[t]):
                toc_broken += 1
                err(f"NAV 목차 링크 오류: {h}")
    res["checked"].update(toc_links=toc_links, toc_broken=toc_broken)
    # cover
    cover = any("cover-image" in m["properties"] for m in book.manifest.values()) or \
        book.opf.find("opf:metadata/opf:meta[@name='cover']", NS) is not None
    res["checked"]["cover_declared"] = bool(cover)
    return res


def find_epubcheck() -> list[str] | None:
    jar = os.environ.get("EPUBCHECK_JAR")
    java = os.environ.get("JAVA_BIN") or shutil.which("java")
    if jar and os.path.exists(jar) and java:
        return [java, "-jar", jar]
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cand = os.path.join(here, "tools", "epubcheck", "epubcheck.jar")
    if os.path.exists(cand) and java:
        return [java, "-jar", cand]
    if shutil.which("epubcheck"):
        return ["epubcheck"]
    return None


def run_epubcheck(path: str, timeout=600) -> dict:
    cmd = find_epubcheck()
    if not cmd:
        return {"available": False, "message": "EPUBCheck 미설치 (EPUBCHECK_JAR, java 필요) - 건너뜀"}
    try:
        import tempfile
        jf = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        jf.close()
        p = subprocess.run(cmd + [path, "--json", jf.name], capture_output=True, text=True, timeout=timeout,
                           encoding="utf-8", errors="replace")
        try:
            data = json.load(open(jf.name, encoding="utf-8"))
            os.unlink(jf.name)
            msgs = data.get("messages", [])
            cnt = Counter(m.get("severity") for m in msgs)
            return {"available": True, "exit_code": p.returncode,
                    "version": data.get("checker", {}).get("checkerVersion"),
                    "fatal": cnt.get("FATAL", 0), "errors": cnt.get("ERROR", 0), "warnings": cnt.get("WARNING", 0),
                    "messages": [{"severity": m.get("severity"), "id": m.get("ID"), "message": m.get("message"),
                                  "locations": [f"{l.get('path')}:{l.get('line')}" for l in (m.get("locations") or [])[:3]]}
                                 for m in msgs][:200]}
        except json.JSONDecodeError:
            return {"available": True, "exit_code": p.returncode, "raw": (p.stdout + p.stderr)[-4000:]}
    except Exception as e:
        return {"available": False, "message": f"EPUBCheck 실행 실패: {e}"}


def translation_check(rows: list[dict], glossary: list[dict], characters: list[dict], tgt_lang="ko") -> dict:
    issues = []
    stat = Counter()
    for r in rows:
        sev, flags = check_block(r["source"], r.get("target"), r["kind"], tgt_lang)
        stat[sev] += 1
        if sev != "ok":
            issues.append({"block_id": r["block_id"], "chapter": r["chapter"], "severity": sev, "flags": flags,
                           "source": strip_tokens(r["source"])[:300], "target": strip_tokens(r.get("target") or "")[:300]})
    dups = find_duplicates(rows)
    for bid, of in dups.items():
        r = next(x for x in rows if x["block_id"] == bid)
        issues.append({"block_id": bid, "chapter": r["chapter"], "severity": "error", "flags": [f"duplicate-of:{of}"],
                       "source": strip_tokens(r["source"])[:300], "target": strip_tokens(r.get("target") or "")[:300]})
    # dialogue preservation (global count)
    src_dlg = sum(1 for r in rows if is_dialogue(r["source"]))
    tgt_dlg = sum(1 for r in rows if is_dialogue(r["source"]) and r.get("target") and target_quote_count(r["target"]) > 0)
    # glossary consistency: when source term occurs, target term should occur
    gl_issues = []
    for g in glossary:
        if not g.get("target") or len(g["source"]) < 2:
            continue
        occ = [r for r in rows if g["source"] in strip_tokens(r["source"]) and r.get("target")]
        miss = [r for r in occ if g["target"] not in r["target"]]
        if occ and miss:
            gl_issues.append({"source": g["source"], "target": g["target"], "occurrences": len(occ),
                              "inconsistent": len(miss), "examples": [m["block_id"] for m in miss[:5]]})
    # character speech: dialogue lines adjacent to a character's name -> register mix (heuristic)
    sp_issues = []
    for c in characters:
        k = c.get("korean")
        if not k:
            continue
        lines = [r for r in rows if r.get("target") and is_dialogue(r["source"]) and k in r["target"]]
        polite = sum(1 for r in lines if re.search(r"(습니다|세요|어요|아요|해요|죠)[.?!…”\s]", r["target"]))
        if len(lines) >= 6 and 0.25 < polite / len(lines) < 0.75:
            sp_issues.append({"character": c["name"], "korean": k, "dialogue_lines": len(lines),
                              "polite_ratio": round(polite / len(lines), 2),
                              "note": "존댓말/반말 혼재 (휴리스틱; 수동 확인 권장)"})
    return {"blocks": len(rows), "ok": stat["ok"], "warn": stat["warn"], "error": stat["error"] + len(dups),
            "untranslated": sum(1 for i in issues if any(f in ("missing", "no-hangul") or f.startswith("residual") for f in i["flags"])),
            "duplicates": len(dups), "dialogue_source": src_dlg, "dialogue_preserved": tgt_dlg,
            "glossary_inconsistencies": gl_issues, "speech_inconsistencies": sp_issues, "issues": issues}


def write_reports(report: dict, json_path: str, html_path: str):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    e = html.escape
    t = report.get("translation", {})
    s = report.get("structure", {})
    ec = report.get("epubcheck", {})
    rows = "".join(
        f"<tr class='{e(i['severity'])}'><td>{e(i['block_id'])}</td><td>{e(i['chapter'] or '')}</td><td>{e(i['severity'])}</td>"
        f"<td>{e(', '.join(i['flags']))}</td><td>{e(i['source'])}</td><td>{e(i['target'])}</td></tr>"
        for i in t.get("issues", [])[:2000])
    gl = "".join(f"<tr><td>{e(g['source'])}</td><td>{e(g['target'])}</td><td>{g['occurrences']}</td><td>{g['inconsistent']}</td><td>{e(', '.join(g['examples']))}</td></tr>"
                 for g in t.get("glossary_inconsistencies", []))
    ecm = "".join(f"<li><b>{e(m['severity'] or '')}</b> {e(m['id'] or '')}: {e(m['message'] or '')} <small>{e(' '.join(m['locations']))}</small></li>"
                  for m in ec.get("messages", []))
    glossary = "".join(f"<tr><td>{e(g['source'])}</td><td>{e(g.get('target') or '')}</td><td>{e(g.get('type') or '')}</td><td>{e(g.get('reading') or '')}</td></tr>"
                       for g in report.get("glossary", []))
    chars = "".join(f"<tr><td>{e(c['name'])}</td><td>{e(c.get('korean') or '')}</td><td>{e(c.get('speech') or '')}</td><td>{e(c.get('honorifics') or '')}</td></tr>"
                    for c in report.get("characters", []))
    j = report.get("job", {})
    doc = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>번역 검수 보고서 - {e(j.get('title') or '')}</title>
<style>body{{font-family:system-ui,'Malgun Gothic',sans-serif;margin:24px;color:#222}}table{{border-collapse:collapse;width:100%;font-size:13px}}
td,th{{border:1px solid #ddd;padding:4px 6px;vertical-align:top}}th{{background:#f3f3f3}}tr.error td{{background:#fff0f0}}tr.warn td{{background:#fffbe6}}
.card{{display:inline-block;border:1px solid #ddd;border-radius:8px;padding:10px 16px;margin:4px}}.ok{{color:#090}}.bad{{color:#c00}}</style></head><body>
<h1>번역 검수 보고서</h1><p><b>{e(j.get('title') or '')}</b> — 엔진 {e(str(j.get('engine')))} / {e(str(j.get('model')))} — 생성 {e(report.get('generated',''))}</p>
<div class="card">전체 블록 <b>{t.get('blocks',0)}</b></div><div class="card">정상 <b class="ok">{t.get('ok',0)}</b></div>
<div class="card">경고 <b>{t.get('warn',0)}</b></div><div class="card">오류 <b class="{'bad' if t.get('error') else 'ok'}">{t.get('error',0)}</b></div>
<div class="card">미번역 <b>{t.get('untranslated',0)}</b></div><div class="card">중복 <b>{t.get('duplicates',0)}</b></div>
<div class="card">대화문 {t.get('dialogue_preserved',0)}/{t.get('dialogue_source',0)}</div>
<div class="card">토큰 in/out {j.get('prompt_tokens',0):,}/{j.get('completion_tokens',0):,}</div>
<div class="card">API 호출 {j.get('api_calls',0)} / 재시도 {j.get('retries',0)}</div>
<div class="card">재번역 라운드 {report.get('retranslation',{}).get('rounds',0)} (블록 {report.get('retranslation',{}).get('blocks',0)})</div>
<h2>1차: 구조 검사 <span class="{'ok' if s.get('ok') else 'bad'}">{'통과' if s.get('ok') else '문제 있음'}</span></h2>
<pre>{e(json.dumps(s.get('checked',{}), ensure_ascii=False, indent=1))}</pre><ul>{''.join(f'<li class=bad>{e(x)}</li>' for x in s.get('errors',[]))}{''.join(f'<li>{e(x)}</li>' for x in s.get('warnings',[])[:100])}</ul>
<h2>EPUBCheck</h2><p>{e(json.dumps({k: v for k, v in ec.items() if k != 'messages'}, ensure_ascii=False))}</p><ul>{ecm}</ul>
<h2>2차: 번역 검사 — 문제 블록</h2><table><tr><th>블록 ID</th><th>장</th><th>심각도</th><th>플래그</th><th>원문</th><th>번역</th></tr>{rows}</table>
<h2>고유명사 불일치</h2><table><tr><th>원어</th><th>용어집</th><th>등장</th><th>불일치</th><th>예시</th></tr>{gl}</table>
<h2>인물 말투 (휴리스틱)</h2><pre>{e(json.dumps(t.get('speech_inconsistencies',[]), ensure_ascii=False, indent=1))}</pre>
<h2>용어집</h2><table><tr><th>원어</th><th>번역</th><th>종류</th><th>읽기</th></tr>{glossary}</table>
<h2>인물 DB</h2><table><tr><th>이름</th><th>한국어</th><th>말투</th><th>호칭</th></tr>{chars}</table>
</body></html>"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(doc)
